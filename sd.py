import psycopg2

conn = psycopg2.connect(
    host="ep-quiet-flower-adkm6iq9-pooler.c-2.us-east-1.aws.neon.tech",
    database="neondb",
    user="neondb_owner",
    password="npg_dTWiUe6JGc4r",  # paste password directly here to test
    sslmode="require"
)
print("Connected!")