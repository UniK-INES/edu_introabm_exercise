'''
Created on 14.05.2025

@author: Sascha Holzhauer
'''
import os

if __name__ == '__main__':
    if "SOLARA_APP" not in os.environ:
        os.environ["SOLARA_APP"] = "app"

    import solara.server.starlette

    server = solara.server.starlette.ServerStarlette(host="localhost", port=50506)
    print(f"Starting server on {server.base_url}")
    server.serve_threaded()
    server.wait_until_serving(timeout=30)
    server.join()