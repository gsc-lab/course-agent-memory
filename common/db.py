"""
db.py — pgvector(Postgres) 연결 헬퍼

gen3 이후 예제는 모두 같은 Postgres/pgvector DB를 사용한다.
DB 연결, 스키마 적용, vector 타입 등록을 이 파일 하나로 처리한다.

[순서가 중요한 이유]
pgvector의 vector 타입이 DB에 먼저 만들어져야 numpy 배열을 저장할 수 있다.
그래서 schema.sql을 적용한 뒤 vector 타입 변환을 등록한다.
"""

import os
import pathlib

from dotenv import load_dotenv

load_dotenv()  # .env의 DATABASE_URL을 환경변수로 불러온다.

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://agent:agent@localhost:5433/agent_memory"
)
SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def connect():
    """pgvector에 연결하고 필요한 테이블을 준비한 커넥션을 돌려준다."""
    import psycopg
    from pgvector.psycopg import register_vector

    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:  # 대부분 DB가 아직 켜지지 않은 경우다.
        raise SystemExit(
            "pgvector에 연결할 수 없습니다. 먼저 `docker compose up -d` 로 DB를 켜주세요.\n"
            f"  DATABASE_URL={DATABASE_URL}\n  원인: {e}"
        )

    # schema.sql을 적용해 extension과 테이블을 준비한다.
    # 그 다음 register_vector로 numpy 배열 <-> DB vector 변환을 등록한다.
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in (s.strip() for s in schema_sql.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()
    register_vector(conn)
    return conn
