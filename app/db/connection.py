from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine: AsyncEngine | None = None
async_session: async_sessionmaker[AsyncSession] | None = None


def init_db() -> None:
    global engine, async_session
    db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_async_engine(db_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)


async def close_db() -> None:
    if engine:
        await engine.dispose()


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if async_session is None:
        raise RuntimeError("Database not initialised")
    return async_session
