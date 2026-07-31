import asyncio
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_query import CodexRag, ask_llm, build_prompt
from settings import BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот по Трудовому кодексу РФ.\nЗадай вопрос."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    rag: CodexRag = context.application.bot_data["rag"]

    await update.message.reply_text("Ищу ответ...")

    try:
        loop = asyncio.get_running_loop()

        # SentenceTransformer.encode и вызовы Groq — блокирующие,
        # поэтому уводим их в executor, чтобы не морозить event loop бота
        output = await loop.run_in_executor(
            None, lambda: rag.search(query=query, use_query_transform=True)
        )
        prompt = build_prompt(output["original_query"], output["results"])
        answer = await loop.run_in_executor(None, lambda: ask_llm(prompt))

        header = (
            f"Ваш вопрос переформулирован как:\n"
            f"«{output['transformed_query']}»\n\n"
        )
        await update.message.reply_text(header + answer)

    except Exception:
        logger.exception("Ошибка обработки запроса")
        await update.message.reply_text("Ошибка обработки запроса, попробуйте ещё раз")


def main():
    logger.info("Загрузка модели и индекса...")
    rag = CodexRag()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.bot_data["rag"] = rag

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()