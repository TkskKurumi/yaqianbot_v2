import platform
import os
if(platform.system() == "Windows"):
    from .typedleveldb_win32 import TypedLevelDB
elif(os.environ.get("PYTHON_LEVELDB_BACKEND") == "plyvel"):
    from .typedleveldb_win32 import TypedLevelDB
else:
    from .typedleveldb import TypedLevelDB