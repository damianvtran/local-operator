from local_operator.local_operator import DeepSeekCLI
import asyncio

if __name__ == "__main__":
    try:
        cli = DeepSeekCLI()
        asyncio.run(cli.chat())
    except Exception as e:
        print(f"Error: {str(e)}")
