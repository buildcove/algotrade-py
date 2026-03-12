if __name__ == "__main__":
    import uvicorn

    uvicorn.run("algotrade.api.app:app", host="127.0.0.1", port=8000, reload=True)
