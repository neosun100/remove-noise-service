#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

# 导入原始API
from api_enhanced import app

# 添加Swagger支持
from api_swagger import add_swagger_routes
add_swagger_routes(app)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5080, threads=8)
