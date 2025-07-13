import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Type
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# MongoDB and other imports
from pymongo import MongoClient
from dotenv import load_dotenv
import re

# === Load config ===
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === MongoDB Connections ===
try:
    client = MongoClient(MONGO_URI)
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful")
    
    dbs = {
        "deficosmos.Borrowers": client["deficosmos"]["Borrowers"],
        "deficosmos.Lenders": client["deficosmos"]["Lenders"],
        "deficosmos.Loans": client["deficosmos"]["Loans"],
        "deficosmos.Installments": client["deficosmos"]["Installments"],
        "deficosmos.Transactions": client["deficosmos"]["Transactions"]
    }
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# === Enhanced Schema Information ===
SCHEMA_INFO = """
Database Schema Information:

Collections and Fields:
1. deficosmos.Borrowers
   - borrower_id (string): Unique identifier for borrower
   - name (string): Borrower's full name
   - phone_number (string): Phone number
   - aadhar_number (string): Aadhaar number (format: ####-####-####)
   - upi_id (string): UPI payment ID
   - created_at (date): Account creation date
   - cibil_score (number): CIBIL credit score (300-900)

2. deficosmos.Lenders
   - lender_id (string): Unique identifier for lender
   - name (string): Lender's full name
   - phone_number (string): Phone number
   - upi_id (string): UPI payment ID
   - created_at (date): Account creation date

3. deficosmos.Loans
   - loan_id (string): Unique loan identifier
   - borrower_id (string): Links to Borrowers collection
   - lender_id (string): Links to Lenders collection
   - principal_amount (number): Loan amount
   - interest_rate (number): Interest rate percentage
   - duration_months (number): Loan duration in months
   - start_date (date): Loan start date
   - status (string): Loan status (Requested, Funded, Repaid, Defaulted)
   - created_at (date): Loan creation date

4. deficosmos.Installments
   - installment_id (string): Unique installment identifier
   - loan_id (string): Links to Loans collection
   - due_date (date): Due date for installment
   - amount (number): Installment amount
   - paid (boolean): Payment status
   - paid_at (date): Payment timestamp (if paid)
   - transaction_id (string): Links to Transactions collection (if paid)

5. deficosmos.Transactions
   - transaction_id (string): Unique transaction identifier
   - from_user_id (string): Source user/entity
   - to_user_id (string): Destination user/entity
   - amount (number): Transaction amount
   - transaction_type (string): Type (Repayment, Disbursement)
   - timestamp (date): Transaction timestamp
   - upi_reference (string): UPI reference number
   - loan_id (string): Associated loan ID

Relationships:
- deficosmos.Borrowers.borrower_id <-> deficosmos.Loans.borrower_id
- deficosmos.Lenders.lender_id <-> deficosmos.Loans.lender_id
- deficosmos.Loans.loan_id <-> deficosmos.Installments.loan_id
- deficosmos.Installments.transaction_id <-> deficosmos.Transactions.transaction_id
- deficosmos.Transactions.loan_id <-> deficosmos.Loans.loan_id

Key Features:
- Track lending relationships between borrowers and lenders
- Monitor loan repayment through installments
- Link transactions to specific loans and installments
- Credit scoring with CIBIL scores for borrowers
- Support for various loan statuses and payment tracking
Can Teresa Perez take a loan of 10000 at 6% for 24 months?"

Answers should be on to the point and crisp.
"""

# === Advanced Query Parser ===
class AdvancedQueryParser:
    def __init__(self):
        self.entity_patterns = {
            'borrowers': ['borrower', 'borrowers', 'customer', 'customers', 'debtor', 'debtors'],
            'lenders': ['lender', 'lenders', 'investor', 'investors', 'creditor', 'creditors'],
            'loans': ['loan', 'loans', 'credit', 'debt', 'lending', 'borrow'],
            'installments': ['installment', 'installments', 'emi', 'payment', 'payments'],
            'transactions': ['transaction', 'transactions', 'transfer', 'transfers', 'tx'],
            'cibil': ['cibil', 'credit score', 'score', 'rating']
        }
        
        self.status_patterns = {
            'requested': ['requested', 'pending', 'applied'],
            'funded': ['funded', 'approved', 'disbursed', 'active'],
            'repaid': ['repaid', 'completed', 'closed', 'finished'],
            'defaulted': ['defaulted', 'failed', 'overdue', 'delinquent'],
            'paid': ['paid', 'settled', 'completed'],
            'unpaid': ['unpaid', 'pending', 'due', 'outstanding']
        }
        
        self.action_patterns = {
            'show': ['show', 'display', 'list', 'get', 'find', 'search', 'fetch'],
            'count': ['count', 'how many', 'number of', 'total'],
            'update': ['update', 'modify', 'change', 'edit'],
            'analyze': ['analyze', 'analysis', 'summary', 'report', 'statistics']
        }
        
        # Enhanced numeric operators
        self.numeric_operators = {
            'greater_than': [
                r'greater than (\d+(?:\.\d+)?)',
                r'more than (\d+(?:\.\d+)?)',
                r'above (\d+(?:\.\d+)?)',
                r'over (\d+(?:\.\d+)?)',
                r'> (\d+(?:\.\d+)?)',
                r'gt (\d+(?:\.\d+)?)'
            ],
            'less_than': [
                r'less than (\d+(?:\.\d+)?)',
                r'below (\d+(?:\.\d+)?)',
                r'under (\d+(?:\.\d+)?)',
                r'< (\d+(?:\.\d+)?)',
                r'lt (\d+(?:\.\d+)?)'
            ],
            'equal_to': [
                r'equal to (\d+(?:\.\d+)?)',
                r'equals (\d+(?:\.\d+)?)',
                r'= (\d+(?:\.\d+)?)',
                r'exactly (\d+(?:\.\d+)?)'
            ],
            'between': [
                r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)',
                r'from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)'
            ]
        }
        
        # Field mappings for numeric operations
        self.numeric_fields = {
            'amount': ['amount', 'principal', 'loan amount', 'principal amount', 'value'],
            'interest': ['interest', 'rate', 'interest rate'],
            'duration': ['duration', 'months', 'term', 'period'],
            'cibil': ['cibil', 'score', 'credit score', 'cibil score']
        }
            
        # Cross-collection patterns
        self.cross_collection_patterns = {
            'borrower_loans': ['borrower.*loan', 'loan.*borrower', 'who.*borrowed', 'loans.*by'],
            'lender_loans': ['lender.*loan', 'loan.*lender', 'who.*lent', 'loans.*to'],
            'loan_installments': ['loan.*installment', 'installment.*loan', 'payment.*schedule'],
            'loan_transactions': ['loan.*transaction', 'transaction.*loan', 'payment.*history'],
            'borrower_cibil': ['borrower.*cibil', 'cibil.*borrower', 'credit.*score'],
            'overdue_payments': ['overdue', 'late.*payment', 'missed.*payment', 'defaulted.*installment']
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into MongoDB operations with enhanced numeric support"""
        query_lower = query.lower()
        
        print(f"🔍 Parsing query: {query}")
        
        # Extract numeric filters first
        numeric_filters = self._extract_numeric_filters(query_lower)
        print(f"📊 Numeric filters found: {numeric_filters}")
        
        # Detect cross-collection queries
        if self._is_cross_collection_query(query_lower):
            result = self._build_cross_collection_query(query_lower)
            # Add numeric filters to cross-collection query
            if numeric_filters:
                if 'filters' not in result:
                    result['filters'] = {}
                result['filters'].update(numeric_filters)
            return result
        
        # Detect action type
        action = self._detect_action(query_lower)
        
        # Build query based on entities and actions
        if any(word in query_lower for word in self.entity_patterns['borrowers']):
            result = self._build_borrower_query(query_lower, action)
        elif any(word in query_lower for word in self.entity_patterns['lenders']):
            result = self._build_lender_query(query_lower, action)
        elif any(word in query_lower for word in self.entity_patterns['loans']):
            result = self._build_loan_query(query_lower, action)
        elif any(word in query_lower for word in self.entity_patterns['installments']):
            result = self._build_installment_query(query_lower, action)
        elif any(word in query_lower for word in self.entity_patterns['transactions']):
            result = self._build_transaction_query(query_lower, action)
        elif any(word in query_lower for word in self.entity_patterns['cibil']):
            result = self._build_cibil_query(query_lower, action)
        else:
            # Default to showing loans
            result = self._build_loan_query(query_lower, action)
        
        # Add numeric filters to the result
        if numeric_filters:
            if 'filter' not in result:
                result['filter'] = {}
            result['filter'].update(numeric_filters)
        
        print(f"📋 Final query plan: {result}")
        return result
    
    def _extract_numeric_filters(self, query: str) -> Dict[str, Any]:
        """Extract numeric filters from query"""
        filters = {}
        
        # Determine which field is being filtered
        field_name = self._detect_numeric_field(query)
        if not field_name:
            return filters
        
        print(f"🎯 Detected field: {field_name}")
        
        # Check for different numeric operators
        for operator, patterns in self.numeric_operators.items():
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    print(f"✅ Found {operator} pattern: {pattern}")
                    if operator == 'greater_than':
                        value = float(match.group(1))
                        filters[field_name] = {"$gt": value}
                        print(f"📈 Setting {field_name} > {value}")
                    elif operator == 'less_than':
                        value = float(match.group(1))
                        filters[field_name] = {"$lt": value}
                        print(f"📉 Setting {field_name} < {value}")
                    elif operator == 'equal_to':
                        value = float(match.group(1))
                        filters[field_name] = value
                        print(f"🎯 Setting {field_name} = {value}")
                    elif operator == 'between':
                        min_val = float(match.group(1))
                        max_val = float(match.group(2))
                        filters[field_name] = {"$gte": min_val, "$lte": max_val}
                        print(f"📊 Setting {field_name} between {min_val} and {max_val}")
                    
                    return filters  # Return first match found
        
        # Fallback to keyword-based filters
        if 'large' in query or 'big' in query or 'high' in query:
            if field_name == 'principal_amount':
                filters[field_name] = {"$gt": 50000}
            elif field_name == 'cibil_score':
                filters[field_name] = {"$gte": 750}
        elif 'small' in query or 'low' in query:
            if field_name == 'principal_amount':
                filters[field_name] = {"$lt": 10000}
            elif field_name == 'cibil_score':
                filters[field_name] = {"$lt": 600}
        
        return filters
    
    def _detect_numeric_field(self, query: str) -> Optional[str]:
        """Detect which numeric field is being referenced"""
        for field, keywords in self.numeric_fields.items():
            for keyword in keywords:
                if keyword in query:
                    # Map to actual MongoDB field names
                    if field == 'amount':
                        return 'principal_amount'
                    elif field == 'interest':
                        return 'interest_rate'
                    elif field == 'duration':
                        return 'duration_months'
                    elif field == 'cibil':
                        return 'cibil_score'
        
        # Default fallback - if no specific field mentioned, assume amount
        if any(op in query for op_list in self.numeric_operators.values() for op in op_list):
            return 'principal_amount'
        
        return None
    
    def _is_cross_collection_query(self, query: str) -> bool:
        """Check if query requires cross-collection operations"""
        for pattern_list in self.cross_collection_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query):
                    return True
        
        # Check for specific phrases that indicate cross-collection needs
        cross_phrases = [
            'borrower.*with.*loan', 'lender.*with.*loan',
            'loan.*payment.*history', 'overdue.*installment',
            'defaulted.*loan.*borrower', 'high.*cibil.*score'
        ]
        
        for phrase in cross_phrases:
            if re.search(phrase, query):
                return True
        
        return False
    
    def _detect_action(self, query: str) -> str:
        """Detect the primary action in the query"""
        for action, patterns in self.action_patterns.items():
            if any(pattern in query for pattern in patterns):
                return action
        return 'show'  # default action
    
    def _build_cross_collection_query(self, query: str) -> Dict[str, Any]:
        """Build cross-collection query operations"""
        
        # Borrower with loans
        if re.search(r'borrower.*loan|loan.*borrower|who.*borrowed', query):
            return {
                "type": "cross_collection",
                "operation": "borrower_loans",
                "description": "Get borrowers with their loan information",
                "collections": ["deficosmos.Borrowers", "deficosmos.Loans"],
                "join_field": "borrower_id",
                "filters": self._extract_status_filters(query)
            }
        
        # Lender with loans
        elif re.search(r'lender.*loan|loan.*lender|who.*lent', query):
            return {
                "type": "cross_collection",
                "operation": "lender_loans",
                "description": "Get lenders with their loan information",
                "collections": ["deficosmos.Lenders", "deficosmos.Loans"],
                "join_field": "lender_id",
                "filters": self._extract_status_filters(query)
            }
        
        # Loan with installments
        elif re.search(r'loan.*installment|installment.*loan|payment.*schedule', query):
            return {
                "type": "cross_collection",
                "operation": "loan_installments",
                "description": "Get loans with their installment details",
                "collections": ["deficosmos.Loans", "deficosmos.Installments"],
                "join_field": "loan_id",
                "filters": self._extract_status_filters(query)
            }
        
        # Overdue payments
        elif re.search(r'overdue|late.*payment|missed.*payment|defaulted.*installment', query):
            return {
                "type": "cross_collection",
                "operation": "overdue_payments",
                "description": "Get overdue installments with borrower details",
                "collections": ["deficosmos.Installments", "deficosmos.Loans", "deficosmos.Borrowers"],
                "join_field": "loan_id",
                "filters": {"paid": False}
            }
        
        # High CIBIL score borrowers
        elif re.search(r'high.*cibil|good.*credit|cibil.*score', query):
            return {
                "type": "cross_collection",
                "operation": "borrower_cibil",
                "description": "Get borrowers with high CIBIL scores",
                "collections": ["deficosmos.Borrowers"],
                "join_field": "borrower_id",
                "filters": {"cibil_score": {"$gte": 750}}
            }
        
        # Default cross-collection query
        return {
            "type": "cross_collection",
            "operation": "general",
            "description": "General cross-collection query",
            "collections": ["deficosmos.Borrowers", "deficosmos.Loans"],
            "join_field": "borrower_id",
            "filters": {}
        }
    
    def _extract_status_filters(self, query: str) -> Dict[str, Any]:
        """Extract status filters from query"""
        filters = {}
        
        # Status filters
        for status, patterns in self.status_patterns.items():
            if any(pattern in query for pattern in patterns):
                if status in ['paid', 'unpaid']:
                    filters['paid'] = status == 'paid'
                else:
                    filters['status'] = status.capitalize()
        
        return filters
    
    def _build_borrower_query(self, query: str, action: str) -> Dict:
        """Build borrower query with enhanced filters"""
        filter_query = {}
        
        # Status filters from existing patterns
        filter_query.update(self._extract_status_filters(query))
        
        return {
            "type": "simple",
            "collection": "deficosmos.Borrowers",
            "filter": filter_query,
            "project": {"_id": 0},
            "action": action
        }
    
    def _build_lender_query(self, query: str, action: str) -> Dict:
        """Build lender query"""
        filter_query = {}
        
        return {
            "type": "simple",
            "collection": "deficosmos.Lenders",
            "filter": filter_query,
            "project": {"_id": 0},
            "action": action
        }
    
    def _build_loan_query(self, query: str, action: str) -> Dict:
        """Build loan query with enhanced filters"""
        filter_query = {}
        
        # Status filters
        filter_query.update(self._extract_status_filters(query))
        
        return {
            "type": "simple",
            "collection": "deficosmos.Loans",
            "filter": filter_query,
            "project": {"_id": 0},
            "action": action
        }
    
    def _build_installment_query(self, query: str, action: str) -> Dict:
        """Build installment query"""
        filter_query = {}
        
        # Payment status filters
        if any(word in query for word in ['paid', 'settled', 'completed']):
            filter_query['paid'] = True
        elif any(word in query for word in ['unpaid', 'pending', 'due', 'outstanding']):
            filter_query['paid'] = False
        
        return {
            "type": "simple",
            "collection": "deficosmos.Installments",
            "filter": filter_query,
            "project": {"_id": 0},
            "action": action
        }
    
    def _build_transaction_query(self, query: str, action: str) -> Dict:
        """Build transaction query"""
        filter_query = {}
        
        # Transaction type filters
        if 'repayment' in query:
            filter_query['transaction_type'] = 'Repayment'
        elif 'disbursement' in query:
            filter_query['transaction_type'] = 'Disbursement'
        
        return {
            "type": "simple",
            "collection": "deficosmos.Transactions",
            "filter": filter_query,
            "project": {"_id": 0},
            "action": action
        }
    
    def _build_cibil_query(self, query: str, action: str) -> Dict:
        """Build CIBIL score query"""
        filter_query = {}
        
        # These will be handled by numeric filters now
        return {
            "type": "simple",
            "collection": "deficosmos.Borrowers",
            "filter": filter_query,
            "project": {"_id": 0},
            "action": action
        }

# === Pydantic Models for Tool Inputs ===
class MongoQueryInput(BaseModel):
    user_query: str = Field(description="Natural language query to execute against MongoDB")

class MongoUpdateInput(BaseModel):
    collection: str = Field(description="Collection name (e.g., 'deficosmos.Borrowers')")
    filter_query: str = Field(description="JSON filter query as string")
    update_data: str = Field(description="JSON update data as string")

class MongoAggregateInput(BaseModel):
    collection: str = Field(description="Collection name")
    pipeline: str = Field(description="MongoDB aggregation pipeline as JSON string")

# === Global Query Parser Instance ===
QUERY_PARSER = AdvancedQueryParser()

# === Enhanced LangChain Tools ===
class EnhancedMongoQueryTool(BaseTool):
    name: str = "mongo_query"
    description: str = """
    Execute complex MongoDB queries for the Deficosmos lending platform using natural language.
    Examples:
    - "Show borrowers with their loan details"
    - "Find borrowers with high CIBIL scores"
    - "List lenders with active loans"
    - "Show overdue installments with borrower details"
    - "Find defaulted loans"
    - "Get borrowers with low credit scores"
    - "Show loan repayment history"
    """
    args_schema: Type[BaseModel] = MongoQueryInput
    
    def _run(self, user_query: str) -> str:
        """Execute enhanced MongoDB query from natural language"""
        try:
            print(f"🔍 Processing query: {user_query}")
            
            # Parse query using global parser instance
            query_plan = QUERY_PARSER.parse_query(user_query)
            print(f"📋 Query plan: {query_plan}")
            
            # Execute based on query type
            if query_plan.get("type") == "cross_collection":
                return self._execute_cross_collection_query(query_plan)
            else:
                return self._execute_simple_query(query_plan)
                
        except Exception as e:
            print(f"❌ Error in _run: {str(e)}")
            return f"❌ Error executing query: {str(e)}"
    
    def _execute_cross_collection_query(self, query_plan: Dict) -> str:
        """Execute cross-collection queries"""
        try:
            operation = query_plan.get("operation", "general")
            
            if operation == "borrower_loans":
                return self._get_borrowers_with_loans(query_plan.get("filters", {}))
            elif operation == "lender_loans":
                return self._get_lenders_with_loans(query_plan.get("filters", {}))
            elif operation == "loan_installments":
                return self._get_loans_with_installments(query_plan.get("filters", {}))
            elif operation == "overdue_payments":
                return self._get_overdue_payments()
            elif operation == "borrower_cibil":
                return self._get_borrowers_by_cibil(query_plan.get("filters", {}))
            else:
                return self._general_cross_collection_query(query_plan)
                
        except Exception as e:
            return f"❌ Error in cross-collection query: {str(e)}"
    
    def _get_borrowers_with_loans(self, filters: Dict) -> str:
        """Get borrowers with their loan information"""
        try:
            borrowers_collection = dbs.get("deficosmos.Borrowers")
            loans_collection = dbs.get("deficosmos.Loans")
            
            if not borrowers_collection or not loans_collection:
                return "❌ Required collections not found"
            
            # Get all borrowers
            borrowers = list(borrowers_collection.find({}, {"_id": 0}))
            
            # For each borrower, get their loans
            results = []
            for borrower in borrowers:
                borrower_id = borrower.get("borrower_id")
                if borrower_id:
                    # Get borrower's loans
                    borrower_loans = list(loans_collection.find(
                        {"borrower_id": borrower_id}, 
                        {"_id": 0}
                    ))
                    
                    if borrower_loans:
                        for loan in borrower_loans:
                            combined_record = {
                                "borrower_name": borrower.get("name"),
                                "borrower_id": borrower.get("borrower_id"),
                                "phone_number": borrower.get("phone_number"),
                                "cibil_score": borrower.get("cibil_score"),
                                "loan_id": loan.get("loan_id"),
                                "principal_amount": loan.get("principal_amount"),
                                "interest_rate": loan.get("interest_rate"),
                                "duration_months": loan.get("duration_months"),
                                "loan_status": loan.get("status"),
                                "start_date": loan.get("start_date")
                            }
                            
                            # Apply filters
                            if filters.get("status"):
                                if loan.get("status") == filters["status"]:
                                    results.append(combined_record)
                            elif filters.get("cibil_score"):
                                cibil_filter = filters["cibil_score"]
                                borrower_cibil = borrower.get("cibil_score", 0)
                                if isinstance(cibil_filter, dict):
                                    if "$gte" in cibil_filter and borrower_cibil >= cibil_filter["$gte"]:
                                        results.append(combined_record)
                                    elif "$lt" in cibil_filter and borrower_cibil < cibil_filter["$lt"]:
                                        results.append(combined_record)
                                else:
                                    results.append(combined_record)
                            else:
                                results.append(combined_record)
            
            if not results:
                return f"🔍 No borrowers with loans found matching the criteria"
            
            return self._format_results(results, f"borrowers with loans ({len(results)} found)")
            
        except Exception as e:
            return f"❌ Error getting borrowers with loans: {str(e)}"
    
    def _get_lenders_with_loans(self, filters: Dict) -> str:
        """Get lenders with their loan information"""
        try:
            lenders_collection = dbs.get("deficosmos.Lenders")
            loans_collection = dbs.get("deficosmos.Loans")
            
            if not lenders_collection or not loans_collection:
                return "❌ Required collections not found"
            
            # Get all lenders
            lenders = list(lenders_collection.find({}, {"_id": 0}))
            
            # For each lender, get their loans
            results = []
            for lender in lenders:
                lender_id = lender.get("lender_id")
                if lender_id:
                    # Get lender's loans
                    lender_loans = list(loans_collection.find(
                        {"lender_id": lender_id}, 
                        {"_id": 0}
                    ))
                    
                    if lender_loans:
                        for loan in lender_loans:
                            combined_record = {
                                "lender_name": lender.get("name"),
                                "lender_id": lender.get("lender_id"),
                                "phone_number": lender.get("phone_number"),
                                "loan_id": loan.get("loan_id"),
                                "principal_amount": loan.get("principal_amount"),
                                "interest_rate": loan.get("interest_rate"),
                                "duration_months": loan.get("duration_months"),
                                "loan_status": loan.get("status"),
                                "start_date": loan.get("start_date")
                            }
                            
                            # Apply filters
                            if filters.get("status"):
                                if loan.get("status") == filters["status"]:
                                    results.append(combined_record)
                            else:
                                results.append(combined_record)
            
            if not results:
                return f"🔍 No lenders with loans found matching the criteria"
            
            return self._format_results(results, f"lenders with loans ({len(results)} found)")
            
        except Exception as e:
            return f"❌ Error getting lenders with loans: {str(e)}"
    
    def _get_loans_with_installments(self, filters: Dict) -> str:
        """Get loans with their installment details"""
        try:
            loans_collection = dbs.get("deficosmos.Loans")
            installments_collection = dbs.get("deficosmos.Installments")
            
            if not loans_collection or not installments_collection:
                return "❌ Required collections not found"
            
            # Get all loans
            loans = list(loans_collection.find({}, {"_id": 0}))
            
            # For each loan, get installments
            results = []
            for loan in loans:
                loan_id = loan.get("loan_id")
                if loan_id:
                    # Get loan's installments
                    loan_installments = list(installments_collection.find(
                        {"loan_id": loan_id}, 
                        {"_id": 0}
                    ))
                    
                    if loan_installments:
                        # Calculate payment statistics
                        total_installments = len(loan_installments)
                        paid_installments = sum(1 for inst in loan_installments if inst.get("paid"))
                        overdue_installments = sum(1 for inst in loan_installments if not inst.get("paid"))
                        
                        combined_record = {
                            "loan_id": loan.get("loan_id"),
                            "principal_amount": loan.get("principal_amount"),
                            "interest_rate": loan.get("interest_rate"),
                            "loan_status": loan.get("status"),
                            "total_installments": total_installments,
                            "paid_installments": paid_installments,
                            "overdue_installments": overdue_installments,
                            "payment_completion": f"{(paid_installments/total_installments)*100:.1f}%"
                        }
                        
                        results.append(combined_record)
            
            if not results:
                return f"🔍 No loans with installments found"
            
            return self._format_results(results, f"loans with installment details ({len(results)} found)")
            
        except Exception as e:
            return f"❌ Error getting loans with installments: {str(e)}"
    
    def _get_overdue_payments(self) -> str:
        """Get overdue installments with borrower details"""
        try:
            installments_collection = dbs.get("deficosmos.Installments")
            loans_collection = dbs.get("deficosmos.Loans")
            borrowers_collection = dbs.get("deficosmos.Borrowers")
            
            if not all([installments_collection, loans_collection, borrowers_collection]):
                return "❌ Required collections not found"
            
            # Get overdue installments
            overdue_installments = list(installments_collection.find(
                {"paid": False}, 
                {"_id": 0}
            ))
            
            if not overdue_installments:
                return "🔍 No overdue installments found"
            
            # For each overdue installment, get loan and borrower details
            results = []
            for installment in overdue_installments:
                loan_id = installment.get("loan_id")
                if loan_id:
                    # Get loan details
                    loan = loans_collection.find_one({"loan_id": loan_id}, {"_id": 0})
                    if loan:
                        borrower_id = loan.get("borrower_id")
                        # Get borrower details
                        borrower = borrowers_collection.find_one({"borrower_id": borrower_id}, {"_id": 0})
                        
                        combined_record = {
                            "borrower_name": borrower.get("name") if borrower else "Unknown",
                            "borrower_phone": borrower.get("phone_number") if borrower else "Unknown",
                            "cibil_score": borrower.get("cibil_score") if borrower else "Unknown",
                            "loan_id": loan.get("loan_id"),
                            "principal_amount": loan.get("principal_amount"),
                            "installment_id": installment.get("installment_id"),
                            "due_date": installment.get("due_date"),
                            "installment_amount": installment.get("amount"),
                            "days_overdue": self._calculate_days_overdue(installment.get("due_date"))
                        }
                        
                        results.append(combined_record)
            
            if not results:
                return "🔍 No overdue payments found with complete details"
            
            return self._format_results(results, f"overdue payments ({len(results)} found)")
            
        except Exception as e:
            return f"❌ Error getting overdue payments: {str(e)}"
    
    def _get_borrowers_by_cibil(self, filters: Dict) -> str:
        """Get borrowers filtered by CIBIL score"""
        try:
            borrowers_collection = dbs.get("deficosmos.Borrowers")
            
            if not borrowers_collection:
                return "❌ Borrowers collection not found"
            
            # Apply CIBIL score filters
            query_filter = {}
            if filters.get("cibil_score"):
                query_filter["cibil_score"] = filters["cibil_score"]
            
            borrowers = list(borrowers_collection.find(query_filter, {"_id": 0}))
            
            if not borrowers:
                return f"🔍 No borrowers found with specified CIBIL score criteria"
            
            return self._format_results(borrowers, f"borrowers by CIBIL score ({len(borrowers)} found)")
            
        except Exception as e:
            return f"❌ Error getting borrowers by CIBIL: {str(e)}"
    
    def _calculate_days_overdue(self, due_date) -> int:
        """Calculate days overdue from due date"""
        try:
            if not due_date:
                return 0
            
            from datetime import datetime
            if isinstance(due_date, str):
                due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            
            days_overdue = (datetime.now() - due_date).days
            return max(0, days_overdue)
        except:
            return 0
    
    def _general_cross_collection_query(self, query_plan: Dict) -> str:
        """Execute general cross-collection queries"""
        try:
            collections = query_plan.get("collections", [])
            join_field = query_plan.get("join_field", "")
            
            if len(collections) < 2:
                return "❌ Insufficient collections for cross-collection query"
            
            # Get primary collection data
            primary_collection = dbs.get(collections[0])
            if not primary_collection:
                return f"❌ Primary collection {collections[0]} not found"
            
            primary_data = list(primary_collection.find({}, {"_id": 0}))
            
            # Join with secondary collections
            results = []
            for record in primary_data:
                join_value = record.get(join_field)
                if join_value:
                    combined_record = record.copy()
                    
                    # Join with each secondary collection
                    for secondary_collection_name in collections[1:]:
                        secondary_collection = dbs.get(secondary_collection_name)
                        if secondary_collection:
                            secondary_data = list(secondary_collection.find(
                                {join_field: join_value}, 
                                {"_id": 0}
                            ))
                            
                            if secondary_data:
                                combined_record[f"{secondary_collection_name.split('.')[-1].lower()}"] = secondary_data
                    
                    results.append(combined_record)
            
            if not results:
                return f"🔍 No results found for cross-collection query"
            
            return self._format_results(results, f"cross-collection results ({len(results)} found)")
            
        except Exception as e:
            return f"❌ Error in general cross-collection query: {str(e)}"
    
    def _execute_simple_query(self, query_plan: Dict) -> str:
        """Execute simple single-collection queries"""
        try:
            collection_name = query_plan.get("collection")
            collection = dbs.get(collection_name)
            
            if not collection:
                return f"❌ Collection {collection_name} not found"
            
            filter_query = query_plan.get("filter", {})
            project_query = query_plan.get("project", {"_id": 0})
            action = query_plan.get("action", "show")
            
            if action == "count":
                count = collection.count_documents(filter_query)
                return f"📊 Count: {count} documents in {collection_name}"
            else:
                # Execute find query
                results = list(collection.find(filter_query, project_query))
                
                if not results:
                    return f"🔍 No results found in {collection_name}"
                
                return self._format_results(results, f"{collection_name} results ({len(results)} found)")
                
        except Exception as e:
            return f"❌ Error in simple query: {str(e)}"
    
    def _format_results(self, results: List[Dict], title: str) -> str:
        """Format query results for display"""
        try:
            if not results:
                return f"🔍 No {title} found"
            
            # Limit results for display
            display_results = results[:10]  # Show first 10 results
            
            formatted_output = f"📋 {title.upper()}\n"
            formatted_output += "=" * 50 + "\n\n"
            
            for i, result in enumerate(display_results, 1):
                formatted_output += f"🔹 Result {i}:\n"
                for key, value in result.items():
                    if isinstance(value, list):
                        formatted_output += f"  {key}: [{len(value)} items]\n"
                    elif isinstance(value, dict):
                        formatted_output += f"  {key}: {json.dumps(value, indent=2)}\n"
                    else:
                        formatted_output += f"  {key}: {value}\n"
                formatted_output += "\n"
            
            if len(results) > 10:
                formatted_output += f"... and {len(results) - 10} more results\n"
            
            return formatted_output
            
        except Exception as e:
            return f"❌ Error formatting results: {str(e)}"

class MongoUpdateTool(BaseTool):
    name: str = "mongo_update"
    description: str = "Update documents in MongoDB collections"
    args_schema: Type[BaseModel] = MongoUpdateInput
    
    def _run(self, collection: str, filter_query: str, update_data: str) -> str:
        """Update documents in MongoDB"""
        try:
            # Get collection
            db_collection = dbs.get(collection)
            if not db_collection:
                return f"❌ Collection {collection} not found"
            
            # Parse JSON strings
            filter_dict = json.loads(filter_query)
            update_dict = json.loads(update_data)
            
            # Execute update
            result = db_collection.update_many(filter_dict, {"$set": update_dict})
            
            return f"✅ Updated {result.modified_count} documents in {collection}"
            
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON: {str(e)}"
        except Exception as e:
            return f"❌ Error updating documents: {str(e)}"

class MongoAggregateTool(BaseTool):
    name: str = "mongo_aggregate"
    description: str = "Execute MongoDB aggregation pipelines for complex analytics"
    args_schema: Type[BaseModel] = MongoAggregateInput
    
    def _run(self, collection: str, pipeline: str) -> str:
        """Execute MongoDB aggregation pipeline"""
        try:
            # Get collection
            db_collection = dbs.get(collection)
            if not db_collection:
                return f"❌ Collection {collection} not found"
            
            # Parse pipeline
            pipeline_dict = json.loads(pipeline)
            
            # Execute aggregation
            results = list(db_collection.aggregate(pipeline_dict))
            
            if not results:
                return f"🔍 No results from aggregation on {collection}"
            
            # Format results
            formatted_output = f"📊 AGGREGATION RESULTS ({len(results)} documents)\n"
            formatted_output += "=" * 50 + "\n\n"
            
            for i, result in enumerate(results[:10], 1):
                formatted_output += f"🔹 Result {i}:\n"
                for key, value in result.items():
                    if key != '_id':  # Skip _id field
                        formatted_output += f"  {key}: {value}\n"
                formatted_output += "\n"
            
            if len(results) > 10:
                formatted_output += f"... and {len(results) - 10} more results\n"
            
            return formatted_output
            
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON in pipeline: {str(e)}"
        except Exception as e:
            return f"❌ Error in aggregation: {str(e)}"

class LoanEligibilityInput(BaseModel):
    borrower_identifier: str = Field(description="Borrower ID or name to check eligibility")
    loan_amount: float = Field(description="Proposed loan amount")
    interest_rate: float = Field(description="Proposed interest rate percentage")
    duration_months: int = Field(description="Proposed loan duration in months")
    currency: str = Field(default="ETH", description="Currency of the loan (e.g., INR, ETH)")

class BorrowerProfile(BaseModel):
            borrower_id: str = Field(description="Unique borrower identifier")
            name: str = Field(description="Borrower's full name")
            cibil_score: int = Field(description="CIBIL credit score (300-900)")
            phone_number: str = Field(description="Borrower's phone number")
            total_loan_amount: float = Field(default=0.0, description="Total outstanding loan amount")
            monthly_installments: float = Field(default=0.0, description="Total monthly installment obligations")
            estimated_income: Optional[float] = Field(default=None, description="Estimated monthly income")

    
class LoanEligibilityTool(BaseTool):
    name: str = "loan_eligibility_check"
    description: str = """
    Check if a proposed loan is suitable for a borrower based on their financial profile.
    Supports lookup by borrower ID or name.
    Example queries:
    - "Is a loan of 50000 at 10% interest for 12 months suitable for borrower with ID B123?"
    - "Can Teresa Perez take a loan of 10000 at 6% for 24 months?"
    """
    args_schema: Type[BaseModel] = LoanEligibilityInput

    def _run(self, borrower_identifier: str, loan_amount: float, interest_rate: float, duration_months: int, currency: str = "INR") -> str:
        """Evaluate loan suitability for a borrower"""
        try:
            # Validate currency
            if currency.upper() == "ETH":
                return "❌ Loans in ETH are not supported. Please use INR or update the schema to include cryptocurrency support."

            # Fetch borrower data
            borrower_collection = dbs.get("deficosmos.Borrowers")
            loans_collection = dbs.get("deficosmos.Loans")
            installments_collection = dbs.get("deficosmos.Installments")

            if not all([borrower_collection, loans_collection, installments_collection]):
                return "❌ Required collections not found"

            # Try lookup by borrower_id first, then by name
            borrower = borrower_collection.find_one({"borrower_id": borrower_identifier}, {"_id": 0})
            if not borrower:
                borrower = borrower_collection.find_one({"name": borrower_identifier}, {"_id": 0})
            if not borrower:
                return f"❌ Borrower with ID or name '{borrower_identifier}' not found"

            # Calculate existing loan obligations
            borrower_id = borrower.get("borrower_id")
            existing_loans = list(loans_collection.find({"borrower_id": borrower_id, "status": {"$in": ["Funded", "Requested"]}}, {"_id": 0}))
            total_loan_amount = sum(loan.get("principal_amount", 0) for loan in existing_loans)
            monthly_installments = 0.0
            for loan in existing_loans:
                loan_id = loan.get("loan_id")
                installments = list(installments_collection.find({"loan_id": loan_id, "paid": False}, {"_id": 0}))
                monthly_installments += sum(install.get("amount", 0) for install in installments) / max(1, loan.get("duration_months", 1))

            # Create borrower profile
            profile = BorrowerProfile(
                borrower_id=borrower_id,
                name=borrower.get("name", "Unknown"),
                cibil_score=borrower.get("cibil_score", 300),
                phone_number=borrower.get("phone_number", "Unknown"),
                total_loan_amount=total_loan_amount,
                monthly_installments=monthly_installments,
                estimated_income=borrower.get("estimated_income", 50000)  # Default or fetch from data
            )

            # Calculate proposed loan EMI
            monthly_rate = interest_rate / (12 * 100)
            emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** duration_months) / ((1 + monthly_rate) ** duration_months - 1)

            # Evaluate eligibility
            eligibility_result = self._evaluate_eligibility(profile, loan_amount, emi, interest_rate, duration_months)
            return self._format_eligibility_result(eligibility_result, profile, loan_amount, interest_rate, duration_months, currency)

        except Exception as e:
            return f"❌ Error evaluating loan eligibility: {str(e)}"

    def _evaluate_eligibility(self, profile: BorrowerProfile, loan_amount: float, emi: float, interest_rate: float, duration_months: int) -> Dict:
        """Evaluate loan eligibility based on financial metrics"""
        result = {"is_eligible": True, "reasons": []}

        # Rule 1: CIBIL score check
        if profile.cibil_score < 600:
            result["is_eligible"] = False
            result["reasons"].append(f"CIBIL score ({profile.cibil_score}) is below minimum threshold of 600")
        elif profile.cibil_score < 700 and interest_rate > 12:
            result["is_eligible"] = False
            result["reasons"].append(f"Interest rate ({interest_rate}%) too high for CIBIL score ({profile.cibil_score})")

        # Rule 2: Debt-to-income ratio (assuming 40% of income for loan repayments)
        total_monthly_obligations = profile.monthly_installments + emi
        dti_ratio = total_monthly_obligations / profile.estimated_income if profile.estimated_income else 1.0
        if dti_ratio > 0.4:
            result["is_eligible"] = False
            result["reasons"].append(f"Debt-to-income ratio ({dti_ratio:.2f}) exceeds 0.4")

        # Rule 3: Total loan amount check
        total_debt = profile.total_loan_amount + loan_amount
        if total_debt > 5 * profile.estimated_income:
            result["is_eligible"] = False
            result["reasons"].append(f"Total debt ({total_debt}) exceeds 5x estimated income")

        # Rule 4: Loan amount cap based on CIBIL
        if profile.cibil_score < 700 and loan_amount > 100000:
            result["is_eligible"] = False
            result["reasons"].append(f"Loan amount ({loan_amount}) too high for CIBIL score ({profile.cibil_score})")

        return result

    def _format_eligibility_result(self, result: Dict, profile: BorrowerProfile, loan_amount: float, interest_rate: float, duration_months: int, currency: str) -> str:
        """Format the eligibility result for display"""
        output = f"📋 LOAN ELIGIBILITY CHECK for Borrower: {profile.name} (ID: {profile.borrower_id})\n"
        output += "=" * 50 + "\n\n"
        output += f"🔹 Borrower: {profile.name}\n"
        output += f"🔹 CIBIL Score: {profile.cibil_score}\n"
        output += f"🔹 Proposed Loan: {currency} {loan_amount:.2f} at {interest_rate}% for {duration_months} months\n"
        output += f"🔹 Existing Debt: {currency} {profile.total_loan_amount:.2f}\n"
        output += f"🔹 Monthly Obligations: {currency} {profile.monthly_installments:.2f}\n\n"

        if result["is_eligible"]:
            output += "✅ Loan is SUITABLE for the borrower\n"
        else:
            output += "❌ Loan is NOT SUITABLE for the borrower\n"
            output += "Reasons:\n"
            for reason in result["reasons"]:
                output += f"  - {reason}\n"

        return output
    
# === Performance Monitoring Tool ===
class MongoPerformanceTool(BaseTool):
    name: str = "mongo_performance"
    description: str = "Monitor MongoDB performance and database statistics"
    args_schema: Type[BaseModel] = MongoQueryInput
    
    def _run(self, user_query: str) -> str:
        """Get MongoDB performance statistics"""
        try:
            stats = {}
            
            # Get database statistics
            db_stats = client["deficosmos"].command("dbStats")
            stats["database"] = {
                "collections": db_stats.get("collections", 0),
                "dataSize": f"{db_stats.get('dataSize', 0) / (1024*1024):.2f} MB",
                "indexSize": f"{db_stats.get('indexSize', 0) / (1024*1024):.2f} MB",
                "objects": db_stats.get("objects", 0)
            }
            
            # Get collection statistics
            collection_stats = {}
            for collection_name in ["Borrowers", "Lenders", "Loans", "Installments", "Transactions"]:
                try:
                    coll_stats = client["deficosmos"].command("collStats", collection_name)
                    collection_stats[collection_name] = {
                        "count": coll_stats.get("count", 0),
                        "size": f"{coll_stats.get('size', 0) / 1024:.2f} KB",
                        "avgObjSize": f"{coll_stats.get('avgObjSize', 0):.2f} bytes"
                    }
                except:
                    collection_stats[collection_name] = {"error": "Collection not found"}
            
            stats["collections"] = collection_stats
            
            # Format output
            formatted_output = "📊 MONGODB PERFORMANCE STATISTICS\n"
            formatted_output += "=" * 50 + "\n\n"
            
            formatted_output += "🗄️ DATABASE OVERVIEW:\n"
            for key, value in stats["database"].items():
                formatted_output += f"  {key}: {value}\n"
            
            formatted_output += "\n📁 COLLECTION STATISTICS:\n"
            for coll_name, coll_stats in stats["collections"].items():
                formatted_output += f"  {coll_name}:\n"
                for key, value in coll_stats.items():
                    formatted_output += f"    {key}: {value}\n"
            
            return formatted_output
            
        except Exception as e:
            return f"❌ Error getting performance stats: {str(e)}"

# === Initialize LangChain Components ===
def initialize_langchain_agent():
    """Initialize LangChain agent with enhanced MongoDB tools"""
    try:
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        # Initialize tools
        tools = [
            EnhancedMongoQueryTool(),
            MongoUpdateTool(),
            MongoAggregateTool(),
            MongoPerformanceTool()
        ]
        
        # Create system prompt
        system_prompt = f"""You are an expert MongoDB database analyst for the Deficosmos lending platform.

{SCHEMA_INFO}

Your capabilities:
1. Execute complex MongoDB queries using natural language
2. Perform cross-collection joins and analysis
3. Update documents when needed
4. Run aggregation pipelines for analytics
5. Monitor database performance
6. Evaluate loan suitability for borrowers based on CIBIL score, debt-to-income ratio, and existing obligations, using either borrower ID or name
7. Generate statistics on suitable loan characteristics (e.g., max loan amount, interest rates, durations) when specific loan parameters are not provided

Guidelines:
- Always use the appropriate tool for each query type
- For cross-collection queries, use the enhanced query capabilities
- For loan eligibility checks, support lookup by borrower ID or name
- For loan statistics, provide maximum loan amounts, acceptable interest rates, and feasible durations
- Provide clear, formatted responses with relevant insights
- When showing results, highlight key patterns and anomalies
- For updates, always confirm the operation was successful
- Be proactive in suggesting related analyses
- If a query involves cryptocurrency (e.g., ETH), inform the user that only INR is supported unless the schema is updated

Example queries you can handle:
- "Show borrowers with their loan details"
- "Find borrowers with high CIBIL scores and their loan performance"
- "List overdue payments with borrower contact information"
- "Get loan repayment statistics by status"
- "Show lenders with the most active loans"
- "Find installments that are overdue by more than 30 days"
- "Can Teresa Perez take a loan of 10000 at 6% for 24 months?"
- "What loan characteristics are suitable for borrower ID 47626b0c-12fd-4d6c-a5ad-d2d5d6344759?"

Always strive to provide actionable insights from the data.
"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        # Create agent executor with memory
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools = [
    EnhancedMongoQueryTool(),
    MongoUpdateTool(),
    MongoAggregateTool(),
    MongoPerformanceTool(),
    LoanEligibilityTool()  # Add the new tool
],
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        return agent_executor
        
    except Exception as e:
        print(f"❌ Error initializing LangChain agent: {e}")
        return None

# === Main Execution ===
def main():
    """Main execution function"""
    print("🚀 Initializing Enhanced MongoDB LangChain Agent for Deficosmos...")
    
    # Initialize agent
    agent = initialize_langchain_agent()
    if not agent:
        print("❌ Failed to initialize agent")
        return
    
    print("✅ Agent initialized successfully!")
    print("💡 Example queries:")
    print("  • 'Show borrowers with their loan details'")
    print("  • 'Find borrowers with high CIBIL scores'")
    print("  • 'List overdue payments with borrower information'")
    print("  • 'Get loan statistics by status'")
    print("  • 'Show database performance statistics'")
    print("\n" + "="*60 + "\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("🤖 Enter your query (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print(f"\n🔄 Processing: {user_input}")
            print("-" * 50)
            
            # Execute query
            start_time = time.time()
            response = agent.invoke({"input": user_input})
            end_time = time.time()
            
            print(f"\n✅ Response (took {end_time - start_time:.2f}s):")
            print(response.get("output", "No output generated"))
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()