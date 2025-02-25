import argparse
import torch
import json
from VectorDB import VectorDB
from utils import load_clip_model_and_processor, batch_image_to_embedding
from time_log import TimeLogger
import os
import sys
import pickle

def main():
    parser = argparse.ArgumentParser(description='Extract data similar to query from vector database.')
    parser.add_argument('--host', default='localhost',
                        help='Host address of the database server (default: localhost)')
    parser.add_argument('--port', type=int, default=9000,
                        help='Port number of the database server (default: 9000)')
    parser.add_argument('--vector_db_name', required=True, help='Name of the vector database')
    # create or update from vectors directly
    parser.add_argument('--online_embedding',type=bool,required=True,help="need to embed inputs online or not")
    parser.add_argument('--vector_file',default=None,help="file path storing vectors, needed when online_embedding is True") 
    parser.add_argument('--image_path_file',default=None,help="file path storing image path corresponding to embeddings,needed when online_embedding is True")
    
    parser.add_argument('--clip_path', default=None, help='Path to the CLIP model and processor')
    parser.add_argument('--dataset_path', default=None, help='Path to the image dataset')
    parser.add_argument('--query_num', type=int, default=50, help='Total number of unique search results')
    parser.add_argument('--k', type=int, default=100, help="finally the size of extracted data is query_num*k, which is deduplicated")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')
    parser.add_argument('--device', default=None, help='Computation device (e.g., "cpu" or "cuda")')
    parser.add_argument('--query_method',default="deduplicated", choices=['deduplicated','deduplicated_adaptive'], required=True,
                        help="Query method: 'deduplicated' for global deduplication search")
    parser.add_argument('--result_file_name', help='json file name for saving results')
    parser.add_argument('--log_file_name', help='file name of time log')
    parser.add_argument('--density_threshold', type=float, default=0.2,
                        help='Minimum valid result ratio')

    args = parser.parse_args()

    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not args.log_file_name:
        args.log_file_name = "time_log_" + args.query_method
    args.log_file_name = "logs/" + args.log_file_name
    time_logger = TimeLogger(log_filename=args.log_file_name, with_current_time=True)
    command_line = "python " + " ".join(sys.argv)
    time_logger.record_text(f"Command: {command_line}")
    if not os.path.exists('results'):
        os.makedirs('results')
    if not args.result_file_name:
        args.result_file_name = "result_" + args.query_method + ".json"
    args.result_file_name = "results/" + args.result_file_name

    vector_db = VectorDB(host=args.host, port=args.port, vector_db_name=args.vector_db_name, device=args.device)
    if(args.online_embedding):
        clip_model, clip_processor = load_clip_model_and_processor(
            args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.clip_path)

    time_logger.record_moment("Model loaded")
    image_paths = []
    for root, _, files in os.walk(args.dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, filename))
    time_logger.start_period("Embedding")
    if(args.online_embedding):
        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor, args.device,
                                                                    args.batch_size)
    else:
        with open(args.vector_file, 'rb') as file:
            image_embeddings = pickle.load(file)
        with open(args.image_path_file, 'rb') as file:
            valid_image_paths = pickle.load(file)
    time_logger.end_period("Embedding")
    final_results = []
    
    if args.query_method == 'deduplicated':
        vector_db.prepare_exclusion_table()
        k = (args.query_num - len(final_results)) // len(image_embeddings)
        # print(f"len(image_embeddings):{len(image_embeddings)}")
        time_logger.start_period("Query")
        while len(final_results) < args.query_num:
            for emb_idx, embedding in enumerate(image_embeddings):
                search_results = vector_db.search_vector_deduplicated(
                    query_vector=embedding,
                    k=k
                )
                final_results.extend(search_results)
                # print(f"emb_idx, len(search_results):{emb_idx}, {len(search_results)}")
                if len(final_results) >= args.query_num:
                    break
            k = k << 1
            # print(f"len(final_result):{len(final_results)}")
            # print(f"k:{k}")
        time_logger.end_period("Query")
    elif args.query_method == 'deduplicated_adaptive':
        collected = set()
        k = max(1, args.query_num // len(image_embeddings))
        k_old = 0
        # print(f"| Initial k: {k}")
        low_density_streak = 0
        should_break = False
        time_logger.start_period("Query")

        while len(collected) < args.query_num and low_density_streak < 3:
            start_count = len(collected)

            for i in range(0, len(image_embeddings), args.batch_size):
                batch_embeddings = image_embeddings[i:i + args.batch_size]

                batch_results = vector_db.search_vectors_adaptive(
                    query_vectors=batch_embeddings,
                    k=k,
                )
                # print(f"batch_results:{batch_results}")
                for res in batch_results:
                    collected.add(res['image_path'])
                    if len(collected) >= args.query_num:
                        should_break = True
                        break
                if should_break:
                    break
            if not should_break:
                current_density = (len(collected)-start_count) / ((k-k_old) * len(image_embeddings))
                # print(f"| Collected this round: {len(collected)} - {start_count} = {len(collected) - start_count}")
                # print(f"| Current density: {current_density:.2f}")

                k_old = k
                if current_density < args.density_threshold:
                    low_density_streak += 1
                    k = k << 2
                else:
                    low_density_streak = 0
                    k = k << 1
                # print(f"| Current k: {k}")
        final_results = [{"image_path": image_path} for image_path in collected]
        # print(f"| len(collected):{len(collected)}")
        time_logger.end_period("Query")
    time_logger.record_text(f"Count of results: {len(final_results)}")
    out_file = args.result_file_name
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    time_logger.record_moment("Results saved")


if __name__ == '__main__':
    main()
