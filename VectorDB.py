import time

from clickhouse_driver import Client
from utils import batch_image_to_embedding
import sqlite3
import json
import torch
import pickle

class VectorDB:
    def __init__(self, host, port, vector_db_name, device=None):
        self.host = host
        self.port = port
        self.client = Client(host=self.host, port=self.port)
        self.client.execute(f"CREATE DATABASE IF NOT EXISTS {vector_db_name}")
        self.client.execute(f"USE {vector_db_name}")
        self.vector_db_name=vector_db_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def create_vector_db(self, clip_model, clip_processor, image_paths, batch_size):
        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)
        self.client = Client(host='localhost', port='9000')
        self.client.execute(f"USE {self.vector_db_name}")
        create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS images_embeddings
                (
                    id UInt64,
                    image_path String,
                    embedding Array(Float32),
                    CONSTRAINT check_length CHECK length(embedding) = {clip_model.config.projection_dim}
                )
                ENGINE = MergeTree
                ORDER BY id;
                """

        current_max_id = 0
        self.client.execute(create_table_sql)

        records_to_insert = []
        for idx, (embedding, img_path) in enumerate(zip(image_embeddings, valid_image_paths), start=1):
            new_id = current_max_id + idx
            if isinstance(embedding, (list, tuple)):
                embedding_list = embedding
            elif isinstance(embedding, torch.Tensor):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            records_to_insert.append((new_id, img_path, embedding_list))
        self._insert_records(records_to_insert, batch_size)
        
    def create_vector_db_direct(self, vector_file,image_path_file, batch_size=100):
        with open(vector_file, 'rb') as file:
            image_embeddings = pickle.load(file)
        with open(image_path_file, 'rb') as file:
            valid_image_paths = pickle.load(file) 
        self.client = Client(host='localhost', port='9000')
        self.client.execute(f"USE {self.vector_db_name}")
        create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS images_embeddings
                (
                    id UInt64,
                    image_path String,
                    embedding Array(Float32),
                )
                ENGINE = MergeTree
                ORDER BY id;
                """

        current_max_id = 0
        self.client.execute(create_table_sql)

        records_to_insert = []
        for idx, (embedding, img_path) in enumerate(zip(image_embeddings, valid_image_paths), start=1):
            new_id = current_max_id + idx
            if isinstance(embedding, (list, tuple)):
                embedding_list = embedding
            elif isinstance(embedding, torch.Tensor):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            records_to_insert.append((new_id, embedding_list))
        self._insert_records(records_to_insert, batch_size)


    def update_vector_db(self, clip_model, clip_processor, image_paths, batch_size=32):

        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)

        self.client = Client(host='localhost', port='9000')
        self.client.execute(f"USE {self.vector_db_name}")
        result = self.client.execute("SELECT max(id) FROM default.images_embeddings")
        current_max_id = result[0][0] if result[0][0] is not None else 0

        records_to_insert = []
        for idx, (embedding, img_path) in enumerate(zip(image_embeddings, valid_image_paths), start=1):
            new_id = current_max_id + idx
            if isinstance(embedding, (list, tuple)):
                embedding_list = embedding
            elif isinstance(embedding, torch.Tensor):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            records_to_insert.append((new_id, img_path, embedding_list))
        self._insert_records(records_to_insert, batch_size)
        
    def update_vector_direct(self, vector_file,image_path_file, batch_size=100):
        with open(vector_file, 'rb') as file:
            image_embeddings = pickle.load(file)
        with open(image_path_file, 'rb') as file:
            valid_image_paths = pickle.load(file)

        self.client = Client(host='localhost', port='9000')
        self.client.execute(f"USE {self.vector_db_name}")
        result = self.client.execute("SELECT max(id) FROM default.images_embeddings")
        current_max_id = result[0][0] if result[0][0] is not None else 0

        records_to_insert = []
        for idx, (embedding, img_path) in enumerate(zip(image_embeddings, valid_image_paths), start=1):
            new_id = current_max_id + idx
            if isinstance(embedding, (list, tuple)):
                embedding_list = embedding
            elif isinstance(embedding, torch.Tensor):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            records_to_insert.append((new_id, img_path, embedding_list))
        self._insert_records(records_to_insert, batch_size)

    def _convert_embedding_to_str(self, embedding):
        """
        将 Python 中的 embedding (List[float] 或 numpy array) 转成 MyScale 可识别的格式,
        如 [0.1, 0.2, 0.3].
        """
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().tolist()
        embedding_str = "[" + ",".join(f"{float(x)}" for x in embedding) + "]"
        return embedding_str

    def prepare_exclusion_table(self):

        create_temp_table_sql = """
            CREATE TEMPORARY TABLE IF NOT EXISTS temp_excluded_ids
            (
                id UInt64
            )
        """
        truncate_temp_table_sql = "TRUNCATE TABLE temp_excluded_ids"

        self.client.execute(create_temp_table_sql)
        self.client.execute(truncate_temp_table_sql)

    def search_vector_individual(self, query_vector, k):

        embedding_str = self._convert_embedding_to_str(query_vector)

        # 对 IP 度量，一般需要 ORDER BY dist DESC；对 L2/Cosine，一般升序
        # if distance_func.lower() == 'ip':
        #     order_clause = "ORDER BY dist DESC"
        # else:
        #     order_clause = "ORDER BY dist"
        order_clause = "ORDER BY dist"

        sql = f"""
            SELECT 
                id, 
                image_path, 
                distance(embedding, {embedding_str}) AS dist
            FROM images_embeddings
            {order_clause}
            LIMIT {k}
        """
        rows = self.client.execute(sql)

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "image_path": row[1],
                "distance": row[2]
            })
        return results

    def search_vector_deduplicated(self, query_vector, k):
        embedding_str = self._convert_embedding_to_str(query_vector)

        # if distance_func.lower() == 'ip':
        #     order_clause = "ORDER BY dist DESC"
        # else:
        #     order_clause = "ORDER BY dist"
        order_clause = "ORDER BY dist"

        sql = f"""
            SELECT
                id,
                image_path,
                distance(embedding, {embedding_str}) AS dist
            FROM images_embeddings
            WHERE id NOT IN (SELECT id FROM temp_excluded_ids)
            {order_clause}
            LIMIT {k}
        """

        sql = f"""
            SELECT
                id,
                image_path,
                distance(embedding, {embedding_str}) AS dist
            FROM images_embeddings AS img
            LEFT JOIN temp_excluded_ids AS excl
            ON img.id = excl.id
            WHERE excl.id IS NULL
            {order_clause}
            LIMIT {k};
        """

        # sql = f"""
        #         SELECT ie.id, ie.image_path, distance(ie.embedding, {embedding_str}) AS dist
        #         FROM {self.vector_db_name}.images_embeddings ie
        #         LEFT JOIN temp_excluded_ids tei ON ie.id = tei.id
        #         WHERE tei.id IS NULL
        #         ORDER BY dist
        #         LIMIT {k}
        #     """
        # sql = f"""
        #       WITH (
        #     SELECT id, image_path, embedding
        #     FROM images_embeddings
        #     WHERE id NOT IN (SELECT id FROM temp_excluded_ids)
        # ) AS filtered_data
        # SELECT
        #     id,
        #     image_path,
        #     distance(embedding, {embedding_str}) AS dist
        # FROM filtered_data
        # {order_clause}
        # LIMIT {k};
        # """

        rows = self.client.execute(sql)

        if rows:
            insert_data = [(int(r[0]),) for r in rows]  # 需要是可迭代的
            insert_sql = "INSERT INTO temp_excluded_ids (id) VALUES"
            self.client.execute(insert_sql, insert_data)

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "image_path": row[1],
                "distance": row[2]
            })
        return results

    def search_vectors_adaptive(self, query_vectors, k):
        sub_queries = []
        emb_strs = [self._convert_embedding_to_str(vec) for vec in query_vectors]
        for idx, emb_str in enumerate(emb_strs):
            sub_query = f"""
                SELECT 
                    {idx} AS query_idx, 
                    id, 
                    image_path, 
                    distance(embedding, {emb_str}) AS dist
                FROM images_embeddings
                ORDER BY dist
                LIMIT {k}
            """
            sub_queries.append(sub_query)

        union_query = " UNION ALL ".join(sub_queries)
        sql = f"""
            SELECT 
                id, image_path, dist 
            FROM (
                {union_query}
            )
            LIMIT {k * len(query_vectors)}
        """
        rows = self.client.execute(sql)
        return [{
                "id": row[0],
                "image_path": row[1],
                "distance": row[2]
        } for row in rows]
    def add_vector_index(self, index_type, index_name, metric_type):
        add_vector_index_sql = f"""
                ALTER TABLE images_embeddings
                ADD VECTOR INDEX {index_name} embedding TYPE {index_type}('metric_type={metric_type}');
            """
        self.client.execute(add_vector_index_sql)

    def delete_vector_index(self):
        delete_vector_index_sql = """
            ALTER TABLE images_embeddings
            DROP ALL VECTOR INDEXES;
        """
        try:
            self.client.execute(delete_vector_index_sql)
        except Exception as e:
            print(f"Error : {e}")

    def wait_for_index_build(self):
        while True:
            result = self.client.execute(
                "SELECT table, name, status FROM system.vector_indices WHERE table = 'images_embeddings'")
            if result:
                status = result[0][2]  # 获取索引的状态
                if status == 'Built':
                    print("Index build completed.")
                    break
                elif status == 'Error':
                    print("Error occurred during index build.")
                    break
            time.sleep(5)

    def optimize_table(self):
        try:
            self.client.execute("OPTIMIZE TABLE images_embeddings FINAL")
        except Exception as e:
            print(f"Error during table optimization: {e}")

    def _insert_records(self, records, batch_size):
        total_records = len(records)
        if total_records == 0:
            return
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            try:
                self.client.execute(
                    f"INSERT INTO {self.vector_db_name}.images_embeddings (id, image_path, embedding) VALUES",
                    batch
                )
            except Exception as e:
                print(f"Error: {e}")
