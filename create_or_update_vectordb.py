import argparse
import sys
import torch
from VectorDB import VectorDB
from utils import load_clip_model_and_processor
from time_log import TimeLogger
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Create or update a vector database.')
    parser.add_argument('action', choices=['create', 'update'], help='Operation type: create or update the vector database')
    parser.add_argument('--host', default='localhost',
                        help='Host address of the database server (default: localhost)')
    parser.add_argument('--port', type=int, default=9000,
                        help='Port number of the database server (default: 9000)')
    parser.add_argument('--vector_db_name', required=True, help='Name of the vector database')
    parser.add_argument('--clip_path', default=None ,help='Path to the clip model and processor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images or vector insertion')
    parser.add_argument('--device', default=None, help='Computation device (e.g., "cpu" or "cuda")')
    parser.add_argument('--dataset_path', default=None, help='Path to the image dataset')
    # create or update from vectors directly
    parser.add_argument('--online_embedding',type=bool,required=True,help="need to embed inputs online or not")
    parser.add_argument('--vector_file',default=None,help="file path storing vectors, needed when online_embedding is True") 
    parser.add_argument('--image_path_file',default=None,help="file path storing image path corresponding to embeddings,needed when online_embedding is True")
    parser.add_argument('--index_type', choices=['ScaNN', 'FLAT', 'MSTG'], default='DEFAULT',
                        help='Type of vector index to create')
    parser.add_argument('--index_name', default='idx', help='Name of the vector index')
    parser.add_argument('--metric_type', choices=['Cosine', 'L2', 'IP'], default='Cosine',
                        help='Metric type for the vector index')

    args = parser.parse_args()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    time_logger = TimeLogger(log_filename="logs/time_log_" + args.action, with_current_time=True)
    command_line = "python " + " ".join(sys.argv)
    time_logger.record_text(f"Command: {command_line}")
    if(args.online_embedding):
        clip_model, clip_processor = load_clip_model_and_processor(args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.clip_path)
    time_logger.record_moment("Model loaded")
    vector_db = VectorDB(host=args.host, port=args.port, vector_db_name=args.vector_db_name, device=args.device)
    image_paths = []
    for root, _, files in os.walk(args.dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, filename))
    if args.action == 'create':
        if(args.online_embedding):
            vector_db.create_vector_db(clip_model, clip_processor, image_paths, args.batch_size)
        else:
            vector_db.create_vector_db_direct(args.vector_file,args.image_path_file,args.batch_size)
        time_logger.record_moment("Vector database created and data inserted")
        vector_db.optimize_table()
        time_logger.record_moment("table optimized")
        vector_db.add_vector_index(args.index_type, args.index_name, args.metric_type)
        time_logger.record_moment("vector index added")
    elif args.action == 'update':
        vector_db.delete_vector_index()
        time_logger.record_moment("vector index deleted")
        if(args.online_embedding):
            vector_db.update_vector_db(clip_model, clip_processor, image_paths, args.batch_size)
        else:
            vector_db.update_vector_db_direct(args.vector_file,args.image_path_file,args.batch_size)
        time_logger.record_moment("Data inserted")
        vector_db.optimize_table()
        time_logger.record_moment("table optimized")
        vector_db.add_vector_index(args.index_type, args.index_name, args.metric_type)
        time_logger.record_moment("vector index added")

    else:
        print(f"Unknown action type: {args.action}")
        sys.exit(1)

if __name__ == '__main__':
    main()
