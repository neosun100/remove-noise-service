"""
Audio Noise Removal API with Swagger Documentation
"""
from flask import Flask, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import json

# Swagger配置
SWAGGER_URL = '/docs'
API_URL = '/swagger.json'

def create_swagger_spec():
    """生成Swagger规范"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Audio Noise Removal API",
            "description": "AI-powered audio noise removal service using ResembleAI's Demucs model",
            "version": "1.0.0",
            "contact": {
                "name": "API Support",
                "url": "https://noise.aws.xin"
            }
        },
        "servers": [
            {
                "url": "https://noise.aws.xin",
                "description": "Production server"
            },
            {
                "url": "http://localhost:5080",
                "description": "Local development server"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "description": "Check if the service is running and model is loaded",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "healthy"},
                                            "model_loaded": {"type": "boolean"},
                                            "gpu_idle_time": {"type": "number"},
                                            "active_tasks": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api": {
                "post": {
                    "summary": "Remove noise from audio",
                    "description": "Upload an audio file and get back a noise-removed version",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "audio": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Audio file to process (WAV, MP3, FLAC, etc.)"
                                        }
                                    },
                                    "required": ["audio"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Processed audio file",
                            "content": {
                                "audio/wav": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - no audio file provided"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/upload_async": {
                "post": {
                    "summary": "Async noise removal",
                    "description": "Submit audio for async processing, returns task ID",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "audio": {
                                            "type": "string",
                                            "format": "binary"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Task created",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "task_id": {"type": "string"},
                                            "status": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/status/{task_id}": {
                "get": {
                    "summary": "Check task status",
                    "description": "Get the status of an async processing task",
                    "parameters": [
                        {
                            "name": "task_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Task ID from upload_async"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Task status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["processing", "completed", "failed"]},
                                            "result_url": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

def add_swagger_routes(app):
    """添加Swagger路由到Flask应用"""
    
    @app.route('/swagger.json')
    def swagger_spec():
        return jsonify(create_swagger_spec())
    
    # Swagger UI配置
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Audio Noise Removal API",
            'defaultModelsExpandDepth': -1
        }
    )
    
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    
    return app
