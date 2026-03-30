import asyncio
from api import analyzer_app, get_config
import pprint

async def check():
    config = get_config("analyzer_70eb89e30dbdc893")
    state = analyzer_app.get_state(config)
    print("ERROR:")
    pprint.pprint(state.values.get("error"))
    print("\nLOGS:")
    pprint.pprint(state.values.get("logs"))
    print("\nSTATE:")
    pprint.pprint(state.values)
    
asyncio.run(check())
