from src.runner import trainEvalRunner

def main():
    trainer = trainEvalRunner()
    trainer.poll_to_run_service()


if __name__ == '__main__':
    main()