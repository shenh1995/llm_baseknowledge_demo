from pymilvus import MilvusClient

client = None
res = None

try:
    client = MilvusClient("./milvus_demo.db")
    client.create_collection(
        collection_name="my_collection",
        dimension=5  # The vectors we will use in this demo has 384 dimensions
    )

    query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                    0.9029438446296592]

    res = client.search(
        collection_name="my_collection",
        data=[query_vector],
    )
finally:
    # 释放内存
    client.release_collection(
        collection_name="my_collection"
    )
    # 关闭连接
    client.close()
