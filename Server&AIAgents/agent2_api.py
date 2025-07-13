# import uvicorn
# from fastapi import FastAPI, Body
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional

# # Import everything from your original agent2.py
# from agent2 import (
#     initialize_langchain_agent,
#     QUERY_PARSER,
#     dbs
# )

# app = FastAPI(title="Deficosmos MongoDB Agent API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Initialize agent once at startup
# agent_executor = initialize_langchain_agent()

# class QueryRequest(BaseModel):
#     input: str

# class EligibilityRequest(BaseModel):
#     borrower_identifier: str
#     loan_amount: float
#     interest_rate: float
#     duration_months: int
#     currency: Optional[str] = "INR"

# @app.post("/query")
# def run_query(request: QueryRequest):
#     """Run a natural language query through the agent."""
#     try:
#         response = agent_executor.invoke({"input": request.input})
#         # Return response in the format expected by frontend
#         return {"response": response.get("output", "No output generated")}
#     except Exception as e:
#         return {"response": f"Error processing query: {str(e)}"}

# @app.post("/loan-eligibility")
# def check_loan_eligibility(request: EligibilityRequest):
#     """Check loan eligibility for a borrower."""
#     try:
#         # Compose the eligibility query as the agent expects
#         query = (
#             f"Can {request.borrower_identifier} take a loan of {request.loan_amount} "
#             f"at {request.interest_rate}% for {request.duration_months} months?"
#         )
#         response = agent_executor.invoke({"input": query})
#         return {"response": response.get("output", "No output generated")}
#     except Exception as e:
#         return {"response": f"Error processing eligibility check: {str(e)}"}

# @app.get("/db-stats")
# def get_db_stats():
#     """Get MongoDB performance statistics."""
#     try:
#         response = agent_executor.invoke({"input": "Show database performance statistics"})
#         return {"response": response.get("output", "No output generated")}
#     except Exception as e:
#         return {"response": f"Error getting database stats: {str(e)}"}

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# # Add a test endpoint for debugging
# @app.get("/test")
# def test_endpoint():
#     return {"message": "Backend is working!", "agent_available": agent_executor is not None}

# if __name__ == "__main__":
#     uvicorn.run("agent2_api:app", host="0.0.0.0", port=8002, reload=True)
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import traceback
import logging

# Import everything from your original agent2.py
from agent2 import (
    initialize_langchain_agent,
    QUERY_PARSER,
    dbs,
    EnhancedMongoQueryTool,
    MongoUpdateTool,
    MongoAggregateTool,
    MongoPerformanceTool,
    LoanEligibilityTool
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deficosmos MongoDB Agent API",
    description="Enhanced MongoDB Agent API with loan eligibility and cross-collection queries",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent once at startup
try:
    agent_executor = initialize_langchain_agent()
    logger.info("✅ Agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize agent: {e}")
    agent_executor = None

# Initialize individual tools for direct access
try:
    mongo_query_tool = EnhancedMongoQueryTool()
    mongo_update_tool = MongoUpdateTool()
    mongo_aggregate_tool = MongoAggregateTool()
    mongo_performance_tool = MongoPerformanceTool()
    loan_eligibility_tool = LoanEligibilityTool()
    logger.info("✅ Individual tools initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize individual tools: {e}")

# Pydantic models for API requests
class QueryRequest(BaseModel):
    input: str

class EligibilityRequest(BaseModel):
    borrower_identifier: str
    loan_amount: float
    interest_rate: float
    duration_months: int
    currency: Optional[str] = "INR"

class UpdateRequest(BaseModel):
    collection: str
    filter_query: Dict[str, Any]
    update_data: Dict[str, Any]

class AggregateRequest(BaseModel):
    collection: str
    pipeline: list

class DirectQueryRequest(BaseModel):
    user_query: str

# Enhanced response models
class APIResponse(BaseModel):
    success: bool
    response: str
    execution_time: Optional[float] = None
    tool_used: Optional[str] = None

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_available": agent_executor is not None,
        "tools_available": all([
            mongo_query_tool is not None,
            mongo_update_tool is not None,
            mongo_aggregate_tool is not None,
            mongo_performance_tool is not None,
            loan_eligibility_tool is not None
        ])
    }

@app.post("/query", response_model=APIResponse)
def run_query(request: QueryRequest):
    """Run a natural language query through the agent with enhanced error handling."""
    import time
    start_time = time.time()
    
    try:
        if not agent_executor:
            raise HTTPException(status_code=503, detail="Agent not available")
        
        logger.info(f"Processing query: {request.input}")
        
        # Process the query through the agent
        response = agent_executor.invoke({"input": request.input})
        
        execution_time = time.time() - start_time
        
        # Extract the output
        output = response.get("output", "No output generated")
        
        logger.info(f"Query processed successfully in {execution_time:.2f}s")
        
        return APIResponse(
            success=True,
            response=output,
            execution_time=execution_time,
            tool_used="langchain_agent"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return APIResponse(
            success=False,
            response=error_msg,
            execution_time=execution_time,
            tool_used="error_handler"
        )

@app.post("/direct-query", response_model=APIResponse)
def direct_mongo_query(request: DirectQueryRequest):
    """Execute direct MongoDB query using the enhanced query tool."""
    import time
    start_time = time.time()
    
    try:
        if not mongo_query_tool:
            raise HTTPException(status_code=503, detail="MongoDB query tool not available")
        
        logger.info(f"Processing direct query: {request.user_query}")
        
        # Use the enhanced mongo query tool directly
        response = mongo_query_tool._run(request.user_query)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Direct query processed successfully in {execution_time:.2f}s")
        
        return APIResponse(
            success=True,
            response=response,
            execution_time=execution_time,
            tool_used="enhanced_mongo_query"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error in direct query: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return APIResponse(
            success=False,
            response=error_msg,
            execution_time=execution_time,
            tool_used="error_handler"
        )

@app.post("/loan-eligibility", response_model=APIResponse)
def check_loan_eligibility(request: EligibilityRequest):
    """Check loan eligibility for a borrower using the dedicated tool."""
    import time
    start_time = time.time()
    
    try:
        if not loan_eligibility_tool:
            raise HTTPException(status_code=503, detail="Loan eligibility tool not available")
        
        logger.info(f"Checking loan eligibility for: {request.borrower_identifier}")
        
        # Use the loan eligibility tool directly
        response = loan_eligibility_tool._run(
            borrower_identifier=request.borrower_identifier,
            loan_amount=request.loan_amount,
            interest_rate=request.interest_rate,
            duration_months=request.duration_months,
            currency=request.currency
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"Loan eligibility check completed in {execution_time:.2f}s")
        
        return APIResponse(
            success=True,
            response=response,
            execution_time=execution_time,
            tool_used="loan_eligibility"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error checking loan eligibility: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return APIResponse(
            success=False,
            response=error_msg,
            execution_time=execution_time,
            tool_used="error_handler"
        )

@app.post("/update", response_model=APIResponse)
def update_documents(request: UpdateRequest):
    """Update documents in MongoDB collections."""
    import time
    start_time = time.time()
    
    try:
        if not mongo_update_tool:
            raise HTTPException(status_code=503, detail="MongoDB update tool not available")
        
        logger.info(f"Updating documents in collection: {request.collection}")
        
        # Convert dicts to JSON strings for the tool
        filter_json = json.dumps(request.filter_query)
        update_json = json.dumps(request.update_data)
        
        response = mongo_update_tool._run(
            collection=request.collection,
            filter_query=filter_json,
            update_data=update_json
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"Update completed in {execution_time:.2f}s")
        
        return APIResponse(
            success=True,
            response=response,
            execution_time=execution_time,
            tool_used="mongo_update"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error updating documents: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return APIResponse(
            success=False,
            response=error_msg,
            execution_time=execution_time,
            tool_used="error_handler"
        )

@app.post("/aggregate", response_model=APIResponse)
def run_aggregation(request: AggregateRequest):
    """Execute MongoDB aggregation pipeline."""
    import time
    start_time = time.time()
    
    try:
        if not mongo_aggregate_tool:
            raise HTTPException(status_code=503, detail="MongoDB aggregate tool not available")
        
        logger.info(f"Running aggregation on collection: {request.collection}")
        
        # Convert pipeline to JSON string for the tool
        pipeline_json = json.dumps(request.pipeline)
        
        response = mongo_aggregate_tool._run(
            collection=request.collection,
            pipeline=pipeline_json
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"Aggregation completed in {execution_time:.2f}s")
        
        return APIResponse(
            success=True,
            response=response,
            execution_time=execution_time,
            tool_used="mongo_aggregate"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error in aggregation: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return APIResponse(
            success=False,
            response=error_msg,
            execution_time=execution_time,
            tool_used="error_handler"
        )

@app.get("/db-stats", response_model=APIResponse)
def get_db_stats():
    """Get MongoDB performance statistics."""
    import time
    start_time = time.time()
    
    try:
        if not mongo_performance_tool:
            raise HTTPException(status_code=503, detail="MongoDB performance tool not available")
        
        logger.info("Getting database statistics")
        
        response = mongo_performance_tool._run("Show database performance statistics")
        
        execution_time = time.time() - start_time
        
        logger.info(f"Database stats retrieved in {execution_time:.2f}s")
        
        return APIResponse(
            success=True,
            response=response,
            execution_time=execution_time,
            tool_used="mongo_performance"
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error getting database stats: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return APIResponse(
            success=False,
            response=error_msg,
            execution_time=execution_time,
            tool_used="error_handler"
        )

@app.get("/collections")
def list_collections():
    """List all available collections and their basic info."""
    try:
        collections_info = {}
        for name, collection in dbs.items():
            try:
                count = collection.count_documents({})
                collections_info[name] = {
                    "document_count": count,
                    "available": True
                }
            except Exception as e:
                collections_info[name] = {
                    "document_count": 0,
                    "available": False,
                    "error": str(e)
                }
        
        return {
            "collections": collections_info,
            "total_collections": len(dbs)
        }
        
    except Exception as e:
        return {"error": f"Error listing collections: {str(e)}"}

@app.get("/tools")
def list_tools():
    """List all available tools and their status."""
    tools_status = {
        "enhanced_mongo_query": mongo_query_tool is not None,
        "mongo_update": mongo_update_tool is not None,
        "mongo_aggregate": mongo_aggregate_tool is not None,
        "mongo_performance": mongo_performance_tool is not None,
        "loan_eligibility": loan_eligibility_tool is not None,
        "langchain_agent": agent_executor is not None
    }
    
    return {
        "tools": tools_status,
        "all_available": all(tools_status.values())
    }

@app.get("/examples")
def get_example_queries():
    """Get example queries for different tools."""
    return {
        "natural_language_queries": [
            "Show borrowers with their loan details",
            "Find borrowers with high CIBIL scores",
            "List overdue payments with borrower information",
            "Get loan statistics by status",
            "Show database performance statistics",
            "Can Teresa Perez take a loan of 10000 at 6% for 24 months?",
            "Show my CIBIL score and latest transactions"
        ],
        "direct_mongo_queries": [
            "borrowers with cibil score greater than 750",
            "loans with status funded",
            "overdue installments with borrower details",
            "lenders with active loans"
        ],
        "loan_eligibility": {
            "borrower_identifier": "B123 or Teresa Perez",
            "loan_amount": 50000,
            "interest_rate": 10.5,
            "duration_months": 24,
            "currency": "INR"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": [
        "/health", "/query", "/direct-query", "/loan-eligibility", 
        "/update", "/aggregate", "/db-stats", "/collections", "/tools", "/examples"
    ]}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "agent2_api:app", 
        host="0.0.0.0", 
        port=8002, 
        reload=True,
        log_level="info"
    )