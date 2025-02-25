This file describes how to extract useful data from Myscale vector database.

#### Create your conda environment for Myscale database

To create a Conda environment with Python 3.10 and install dependencies using the `requirements.txt` file, follow these steps:

Creating a Conda Environment

```bash
conda create -n {your_env_name} python=3.10
```

Activating the Environment

```bash
conda activate {your_envs_name}
```

Installing Dependencies from requirements.txt

```bash
pip install -r requirements.txt
```

#### Create Myscale Database and Insert Vector Data

To create a Myscale database and insert your vector data, use the following Python command provided in the code:

```bash
python create_or_update_vectordb.py [action] [arguments]
```

##### Explanation of Parameters:

1. **`action`**:
   - **Choices**: `create` (create a new database) or `update` (update an existing database)
   - **Example**: `create`

2. **`--host`**:
   - **Default**: `localhost`
   - **Description**: The host address of the database server.
   - **Example**: `--host="localhost"`

3. **`--port`**:
   - **Default**: `9000`
   - **Description**: The port number of the database server.
   - **Example**: `--port=9000`

4. **`--vector_db_name`**:
   - **Required**: Yes
   - **Description**: The name of the vector database.
   - **Example**: `--vector_db_name="my_vector_db"`

5. **`--batch_size`**:
   - **Default**: `32`
   - **Description**: The batch size for processing images or vectors.
   - **Example**: `--batch_size=32`

6. **`--device`**:
   - **Default**: Automatically detected (CPU or GPU)
   - **Description**: The computation device (e.g., `"cpu"` or `"cuda"`).
   - **Example**: `--device="cuda"`

7. **`--dataset_path`**:
   - **Default**: `None`
   - **Description**: The path to the image dataset (only required if `--online_embedding` is `True`).
   - **Example**: `--dataset_path="/path/to/your/images"`

8. **`--online_embedding`**:
   - **Choices**: `True` or `False`
   - **Description**: Whether to embed inputs online or not (i.e., calculate vectors on-the-fly).
   - **Example**: `--online_embedding=True`

9. **`--vector_file`**:
   - **Default**: `None`
   - **Description**: The file path storing vectors (required if `--online_embedding` is `False`).
   - **Example**: `--vector_file="/path/to/your/vectors.npy"`

10. **`--image_path_file`**:
    - **Default**: `None`
    - **Description**: The file path storing image paths corresponding to embeddings (required if `--online_embedding` is `False`).
    - **Example**: `--image_path_file="/path/to/your/image_paths.txt"`

11. **`--index_type`**:
    - **Choices**: `ScaNN`, `FLAT`, or `MSTG`
    - **Default**: `DEFAULT`
    - **Description**: The type of vector index to create.
    - **Example**: `--index_type=FLAT`

12. **`--index_name`**:
    - **Default**: `idx`
    - **Description**: The name of the vector index.
    - **Example**: `--index_name="idx"`

13. **`--metric_type`**:
    - **Choices**: `Cosine`, `L2`, or `IP`
    - **Default**: `Cosine`
    - **Description**: The metric type for the vector index.
    - **Example**: `--metric_type=L2`

##### Example Commands:

1. **Create a Database with Online Embedding**

Use this command to create a new vector database, embedding images online and inserting the vectors directly into the database:

```bash
python create_or_update_vectordb.py create \
    --host=localhost \
    --port=9000 \
    --vector_db_name=my_vector_db \
    --device=cuda \
    --dataset_path=/path/to/your/images \
    --online_embedding=True \
    --index_type=FLAT \
    --metric_type=Cosine
```

2. **Create a Database from Precomputed Vectors**

Use this command to create a new vector database by inserting precomputed vectors from a file, without online embedding:

```bash
python create_or_update_vectordb.py create \
    --host=localhost \
    --port=9000 \
    --vector_db_name=my_vector_db \
    --vector_file=/path/to/vectors.npy \
    --image_path_file=/path/to/image_paths.txt \
    --online_embedding=False \
    --index_type=FLAT \
    --metric_type=Cosine
```

3. **Update a Database with Online Embedding**

Use this command to update an existing vector database by embedding new images online and appending the vectors to the database:

```bash
python create_or_update_vectordb.py update \
    --host=localhost \
    --port=9000 \
    --vector_db_name=my_vector_db \
    --device=cuda \
    --dataset_path=/path/to/new_images \
    --online_embedding=True \
    --index_type=FLAT \
    --metric_type=Cosine
```

4. **Update a Database from Precomputed Vectors**

Use this command to update an existing vector database by inserting new precomputed vectors from a file:

```bash
python create_or_update_vectordb.py update \
    --host=localhost \
    --port=9000 \
    --vector_db_name=my_vector_db \
    --vector_file=/path/to/new_vectors.npy \
    --image_path_file=/path/to/new_image_paths.txt \
    --online_embedding=False \
    --index_type=FLAT \
    --metric_type=Cosine
```

#### Extract Data from the Vector Database

To extract data related to query vectors from the database, use the following Python command provided in the code:

```bash
python extract_data.py [arguments]
```

##### Key Parameters:

1. **`--query_method`**:
   - **Choices**: `deduplicated` or `deduplicated_adaptive`
   - **Description**: The method used to query the database. 
   - **Example**: `--query_method=deduplicated`
2. **`--query_num`**:
   - **Default**: `50`
   - **Description**: Total number of unique search results. To get decent amount of wanted data, you may offer more than just one query.
   - **Example**: `--query_num=100`
3. **`--k`**:
   - **Default**: `100`
   - **Description**: Number of results per query (final extracted data size is `query_num * k`, deduplicated).
   - **Example**: `--k=100`
4. **`--result_file_name`**:
   - **Default**: `"results/result_{query_method}.json"`
   - **Description**: File name to save the query results (JSON format).
   - **Example**: `--result_file_name="results/my_query_results.json"`
5. **`--log_file_name`**:
   - **Default**: `"logs/time_log_{query_method}"`
   - **Description**: File name for time logging.
   - **Example**: `--log_file_name="logs/custom_time_log"`
6. **`--density_threshold`**:
   - **Default**: `0.2`
   - **Description**: Minimum valid result ratio for adaptive queries.
   - **Example**: `--density_threshold=0.15`
7. **`--batch_size`**:
   - **Default**: `32`
   - **Description**: Batch size for processing images or vectors.
9. **`--online_embedding`**:
   - **Required**: Yes (Boolean)
   - **Description**: Whether to embed query images online.

##### Example Commands:

1. **Extract Data Using Online Embedding**

```bash
python extract_data.py \
    --host=localhost \
    --port=9000 \
    --vector_db_name=my_vector_db \
    --online_embedding=True \
    --dataset_path=/path/to/images \
    --query_method=deduplicated \
    --query_num=100 \
    --k=100 \
    --result_file_name="results/online_query_results.json"
```

2. **Extract Data Using Precomputed Vectors (Offline Embedding)**

```bash
python extract_data.py \
    --host=localhost \
    --port=9000 \
    --vector_db_name=my_vector_db \
    --online_embedding=False \
    --vector_file=/path/to/vectors.npy \
    --image_path_file=/path/to/image_paths.txt \
    --query_method=deduplicated_adaptive \
    --query_num=100 \
    --k=100 \
    --density_threshold=0.3 \
    --result_file_name="results/offline_query_results.json"
```

##### Notes:
- The `--dataset_path` parameter is required when `--online_embedding=True`.
- The `--vector_file` and `--image_path_file` parameters are required when `--online_embedding=False`.
- The `--density_threshold` parameter is used with the `deduplicated_adaptive` query method to control the search density.

- **deduplicated** is a query method that performs a global deduplication search by examining a fixed number of results per query and incrementally increasing the query depth until the required number of unique results is achieved.

- **deduplicated_adaptive** is a query method that dynamically adjusts the number of results per query based on the density of valid results, optimizing the search efficiency and adaptability to varying data densities.