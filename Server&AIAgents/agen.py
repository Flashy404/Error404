import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pymongo import MongoClient
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import uvicorn

class LoanAnalysisAPI:
    def __init__(self, mongodb_url: str, gemini_api_key: str):
        """
        Initialize the Loan Analysis API
        
        Args:
            mongodb_url: MongoDB connection URL
            gemini_api_key: Google Gemini API key
        """
        self.client = MongoClient(mongodb_url)
        self.db = self.client['deficosmos']
        
        # Initialize Gemini AI
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Collections
        self.borrowers = self.db['Borrowers']
        self.loans = self.db['Loans']
        self.installments = self.db['Installments']
        self.transactions = self.db['Transactions']
    
    def get_borrower_complete_data(self, borrower_id: str) -> Dict[str, Any]:
        """
        Get complete borrower data including all loans, installments, and transactions
        
        Args:
            borrower_id: Unique identifier for the borrower
            
        Returns:
            Dictionary containing all borrower data
        """
        try:
            # Get borrower info
            borrower = self.borrowers.find_one({"borrower_id": borrower_id})
            if not borrower:
                return {"error": "Borrower not found"}
            
            # Get all loans for borrower
            loans = list(self.loans.find({"borrower_id": borrower_id}))
            
            loan_data = []
            total_principal = 0
            total_paid = 0
            total_pending = 0
            total_fragments = 0
            paid_fragments = 0
            
            for loan in loans:
                loan_id = loan['loan_id']
                
                # Get installments (loan fragments)
                installments = list(self.installments.find({"loan_id": loan_id}))
                
                # Get transactions for this loan
                transactions = list(self.transactions.find({"loan_id": loan_id}))
                
                # Calculate loan metrics
                loan_principal = loan['principal_amount']
                loan_paid = sum(inst['amount'] for inst in installments if inst['paid'])
                loan_pending = sum(inst['amount'] for inst in installments if not inst['paid'])
                
                total_principal += loan_principal
                total_paid += loan_paid
                total_pending += loan_pending
                total_fragments += len(installments)
                paid_fragments += sum(1 for inst in installments if inst['paid'])
                
                loan_info = {
                    "loan_id": loan_id,
                    "principal_amount": loan_principal,
                    "interest_rate": loan['interest_rate'],
                    "duration_months": loan['duration_months'],
                    "status": loan['status'],
                    "start_date": loan['start_date'].strftime('%Y-%m-%d'),
                    "fragments_total": len(installments),
                    "fragments_paid": sum(1 for inst in installments if inst['paid']),
                    "fragments_pending": sum(1 for inst in installments if not inst['paid']),
                    "amount_paid": loan_paid,
                    "amount_pending": loan_pending,
                    "installments": [
                        {
                            "installment_id": inst['installment_id'],
                            "due_date": inst['due_date'].strftime('%Y-%m-%d'),
                            "amount": inst['amount'],
                            "paid": inst['paid'],
                            "paid_date": inst['paid_at'].strftime('%Y-%m-%d') if inst.get('paid_at') else None
                        }
                        for inst in installments
                    ],
                    "transactions": [
                        {
                            "transaction_id": txn['transaction_id'],
                            "amount": txn['amount'],
                            "type": txn['transaction_type'],
                            "date": txn['timestamp'].strftime('%Y-%m-%d'),
                            "upi_reference": txn['upi_reference']
                        }
                        for txn in transactions
                    ]
                }
                loan_data.append(loan_info)
            
            # Compile final data
            complete_data = {
                "borrower_info": {
                    "borrower_id": borrower['borrower_id'],
                    "name": borrower['name'],
                    "phone": borrower['phone_number'],
                    "cibil_score": borrower['cibil_score'],
                    "created_date": borrower['created_at'].strftime('%Y-%m-%d')
                },
                "portfolio_summary": {
                    "total_loans": len(loans),
                    "total_principal": total_principal,
                    "total_paid": total_paid,
                    "total_pending": total_pending,
                    "total_fragments": total_fragments,
                    "paid_fragments": paid_fragments,
                    "pending_fragments": total_fragments - paid_fragments,
                    "payment_completion_rate": round((paid_fragments / total_fragments * 100), 2) if total_fragments > 0 else 0
                },
                "loans": loan_data
            }
            
            return complete_data
            
        except Exception as e:
            return {"error": f"Error retrieving data: {str(e)}"}
    
    def analyze_loan_performance(self, data: Dict[str, Any]) -> str:
        """
        Analyze loan performance using Gemini AI
        
        Args:
            data: Complete borrower data
            
        Returns:
            Analysis report as string
        """
        try:
            # Calculate payment patterns and spacing
            payment_patterns = self.calculate_payment_patterns(data)
            
            # Create detailed analysis data
            analysis_data = {
                "borrower_cibil": data['borrower_info']['cibil_score'],
                "total_loans": data['portfolio_summary']['total_loans'],
                "total_amount": data['portfolio_summary']['total_principal'],
                "payment_rate": data['portfolio_summary']['payment_completion_rate'],
                "fragments_paid": data['portfolio_summary']['paid_fragments'],
                "fragments_pending": data['portfolio_summary']['pending_fragments'],
                "loan_statuses": [loan['status'] for loan in data['loans']],
                "payment_patterns": payment_patterns
            }
            
            prompt = f"""
            Analyze this borrower loan data and provide a comprehensive assessment:
            
            Borrower Profile:
            - CIBIL Score: {analysis_data['borrower_cibil']}
            - Total Loans: {analysis_data['total_loans']}
            - Total Amount: Rs {analysis_data['total_amount']}
            - Payment Completion Rate: {analysis_data['payment_rate']}%
            
            Fragment Analysis:
            - Paid Fragments: {analysis_data['fragments_paid']}
            - Pending Fragments: {analysis_data['fragments_pending']}
            - Loan Statuses: {analysis_data['loan_statuses']}
            
            Payment Pattern Analysis:
            - Average Payment Gap: {payment_patterns['avg_payment_gap']} days
            - Payment Consistency: {payment_patterns['consistency_score']}
            - Early Payments: {payment_patterns['early_payments']}
            - Late Payments: {payment_patterns['late_payments']}
            - Payment Frequency: {payment_patterns['payment_frequency']}
            - Recent Payment Trend: {payment_patterns['recent_trend']}
            
            Provide detailed analysis in exactly 6 sections:
            1. Risk Assessment - Evaluate overall credit risk based on CIBIL score and payment behavior
            2. Payment Behavior - Analyze payment timing patterns and consistency
            3. Fragment Management - How well the borrower handles loan fragments and installments
            4. Payment Spacing Analysis - Assess gaps between payments and regularity
            5. Portfolio Health - Overall loan portfolio performance and status
            6. Recommendations - Specific actionable recommendations for the borrower
            
            Each section should be 2-3 sentences with clear insights. Use simple language without special characters.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"
    
    def calculate_payment_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate payment patterns and spacing metrics
        
        Args:
            data: Complete borrower data
            
        Returns:
            Dictionary with payment pattern metrics
        """
        try:
            all_payments = []
            total_early = 0
            total_late = 0
            total_on_time = 0
            
            for loan in data['loans']:
                for installment in loan['installments']:
                    if installment['paid'] and installment['paid_date']:
                        payment_info = {
                            'due_date': datetime.strptime(installment['due_date'], '%Y-%m-%d'),
                            'paid_date': datetime.strptime(installment['paid_date'], '%Y-%m-%d'),
                            'amount': installment['amount']
                        }
                        all_payments.append(payment_info)
                        
                        # Calculate early/late payments
                        days_diff = (payment_info['paid_date'] - payment_info['due_date']).days
                        if days_diff < 0:
                            total_early += 1
                        elif days_diff > 0:
                            total_late += 1
                        else:
                            total_on_time += 1
            
            if not all_payments:
                return {
                    'avg_payment_gap': 0,
                    'consistency_score': 'No payments recorded',
                    'early_payments': 0,
                    'late_payments': 0,
                    'payment_frequency': 'No data',
                    'recent_trend': 'No recent activity'
                }
            
            # Sort payments by date
            all_payments.sort(key=lambda x: x['paid_date'])
            
            # Calculate payment gaps
            payment_gaps = []
            for i in range(1, len(all_payments)):
                gap = (all_payments[i]['paid_date'] - all_payments[i-1]['paid_date']).days
                payment_gaps.append(gap)
            
            avg_gap = sum(payment_gaps) / len(payment_gaps) if payment_gaps else 0
            
            # Calculate consistency score
            if len(payment_gaps) > 1:
                gap_variance = sum((gap - avg_gap) ** 2 for gap in payment_gaps) / len(payment_gaps)
                consistency_score = 'High' if gap_variance < 100 else 'Medium' if gap_variance < 400 else 'Low'
            else:
                consistency_score = 'Insufficient data'
            
            # Payment frequency analysis
            if avg_gap <= 7:
                frequency = 'Weekly'
            elif avg_gap <= 15:
                frequency = 'Bi-weekly'
            elif avg_gap <= 35:
                frequency = 'Monthly'
            else:
                frequency = 'Irregular'
            
            # Recent trend analysis
            recent_payments = [p for p in all_payments if (datetime.now() - p['paid_date']).days <= 90]
            if len(recent_payments) >= 2:
                recent_gaps = []
                for i in range(1, len(recent_payments)):
                    gap = (recent_payments[i]['paid_date'] - recent_payments[i-1]['paid_date']).days
                    recent_gaps.append(gap)
                recent_avg = sum(recent_gaps) / len(recent_gaps)
                
                if recent_avg < avg_gap * 0.8:
                    recent_trend = 'Improving frequency'
                elif recent_avg > avg_gap * 1.2:
                    recent_trend = 'Declining frequency'
                else:
                    recent_trend = 'Stable pattern'
            else:
                recent_trend = 'Limited recent activity'
            
            return {
                'avg_payment_gap': round(avg_gap, 1),
                'consistency_score': consistency_score,
                'early_payments': total_early,
                'late_payments': total_late,
                'payment_frequency': frequency,
                'recent_trend': recent_trend
            }
        
        except Exception as e:
            return {
                'avg_payment_gap': 0,
                'consistency_score': 'Error calculating patterns',
                'early_payments': 0,
                'late_payments': 0,
                'payment_frequency': 'Error',
                'recent_trend': 'Error'
            }
            
    def calculate_detailed_payment_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive payment analysis with delay metrics
        
        Args:
            data: Complete borrower data
            
        Returns:
            Dictionary with detailed payment metrics
        """
        try:
            current_date = datetime.now()
            all_payments = []
            overdue_installments = []
            
            on_time_count = 0
            early_count = 0
            late_count = 0
            total_delay_days = 0
            max_delay = 0
            
            for loan in data['loans']:
                for installment in loan['installments']:
                    due_date = datetime.strptime(installment['due_date'], '%Y-%m-%d')
                    
                    if installment['paid'] and installment['paid_date']:
                        paid_date = datetime.strptime(installment['paid_date'], '%Y-%m-%d')
                        delay_days = (paid_date - due_date).days
                        
                        payment_info = {
                            'due_date': due_date,
                            'paid_date': paid_date,
                            'amount': installment['amount'],
                            'delay_days': delay_days,
                            'loan_id': loan['loan_id']
                        }
                        all_payments.append(payment_info)
                        
                        if delay_days < 0:
                            early_count += 1
                        elif delay_days == 0:
                            on_time_count += 1
                        else:
                            late_count += 1
                            total_delay_days += delay_days
                            max_delay = max(max_delay, delay_days)
                    
                    elif not installment['paid'] and due_date < current_date:
                        # Overdue installment
                        overdue_days = (current_date - due_date).days
                        overdue_installments.append({
                            'amount': installment['amount'],
                            'overdue_days': overdue_days,
                            'due_date': due_date,
                            'loan_id': loan['loan_id']
                        })
            
            # Calculate averages and metrics
            total_payments = len(all_payments)
            avg_delay = total_delay_days / late_count if late_count > 0 else 0
            
            # Payment discipline score (0-100)
            if total_payments > 0:
                discipline_score = ((on_time_count + early_count) / total_payments) * 100
            else:
                discipline_score = 0
            
            # Payment gaps analysis
            payment_gaps = []
            if len(all_payments) > 1:
                sorted_payments = sorted(all_payments, key=lambda x: x['paid_date'])
                for i in range(1, len(sorted_payments)):
                    gap = (sorted_payments[i]['paid_date'] - sorted_payments[i-1]['paid_date']).days
                    payment_gaps.append(gap)
            
            avg_gap = sum(payment_gaps) / len(payment_gaps) if payment_gaps else 0
            
            # Consistency rating
            if len(payment_gaps) > 1:
                gap_variance = sum((gap - avg_gap) ** 2 for gap in payment_gaps) / len(payment_gaps)
                if gap_variance < 25:
                    consistency = "Very Consistent"
                elif gap_variance < 100:
                    consistency = "Consistent"
                elif gap_variance < 400:
                    consistency = "Moderately Consistent"
                else:
                    consistency = "Inconsistent"
            else:
                consistency = "Insufficient Data"
            
            # Payment frequency pattern
            if avg_gap <= 7:
                frequency = "Weekly"
            elif avg_gap <= 15:
                frequency = "Bi-weekly"
            elif avg_gap <= 35:
                frequency = "Monthly"
            elif avg_gap <= 70:
                frequency = "Bi-monthly"
            else:
                frequency = "Irregular"
            
            # Recent trend analysis (last 90 days)
            recent_payments = [p for p in all_payments if (current_date - p['paid_date']).days <= 90]
            if len(recent_payments) >= 3:
                recent_delays = [p['delay_days'] for p in recent_payments]
                avg_recent_delay = sum(recent_delays) / len(recent_delays)
                
                if avg_recent_delay < avg_delay * 0.7:
                    recent_trend = "Improving Payment Timing"
                elif avg_recent_delay > avg_delay * 1.3:
                    recent_trend = "Declining Payment Timing"
                else:
                    recent_trend = "Stable Payment Pattern"
            else:
                recent_trend = "Limited Recent Activity"
            
            # Risk assessment
            overdue_amount = sum(inst['amount'] for inst in overdue_installments)
            longest_overdue = max([inst['overdue_days'] for inst in overdue_installments]) if overdue_installments else 0
            
            # Risk level calculation
            if longest_overdue > 90:
                risk_level = "High Risk"
            elif longest_overdue > 30:
                risk_level = "Medium Risk"
            elif late_count > total_payments * 0.3:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"
            
            # Reliability rating
            if discipline_score >= 90:
                reliability = "Excellent"
            elif discipline_score >= 75:
                reliability = "Good"
            elif discipline_score >= 60:
                reliability = "Fair"
            else:
                reliability = "Poor"
            
            # Find next due date and amount
            next_due_date = None
            next_due_amount = 0
            monthly_obligation = 0
            
            for loan in data['loans']:
                for installment in loan['installments']:
                    if not installment['paid']:
                        due_date = datetime.strptime(installment['due_date'], '%Y-%m-%d')
                        if due_date >= current_date:
                            if next_due_date is None or due_date < next_due_date:
                                next_due_date = due_date
                                next_due_amount = installment['amount']
                        
                        # Calculate monthly obligation (next 30 days)
                        if due_date <= current_date + timedelta(days=30):
                            monthly_obligation += installment['amount']
            
            # Last payment analysis
            last_payment_date = None
            if all_payments:
                last_payment_date = max(all_payments, key=lambda x: x['paid_date'])['paid_date']
            
            days_since_last_payment = (current_date - last_payment_date).days if last_payment_date else None
            
            return {
                'on_time_count': on_time_count,
                'early_count': early_count,
                'late_count': late_count,
                'avg_delay_days': round(avg_delay, 1),
                'max_delay_days': max_delay,
                'discipline_score': round(discipline_score, 1),
                'avg_payment_gap': round(avg_gap, 1),
                'consistency_rating': consistency,
                'frequency_pattern': frequency,
                'recent_trend': recent_trend,
                'overdue_count': len(overdue_installments),
                'overdue_amount': overdue_amount,
                'longest_overdue_days': longest_overdue,
                'risk_level': risk_level,
                'reliability_rating': reliability,
                'next_due_date': next_due_date.strftime('%Y-%m-%d') if next_due_date else None,
                'next_due_amount': next_due_amount,
                'monthly_obligation': monthly_obligation,
                'last_payment_date': last_payment_date.strftime('%Y-%m-%d') if last_payment_date else None,
                'days_since_last_payment': days_since_last_payment
            }
            
        except Exception as e:
            return {
                'error': f"Error calculating payment summary: {str(e)}",
                'on_time_count': 0,
                'early_count': 0,
                'late_count': 0,
                'avg_delay_days': 0,
                'max_delay_days': 0,
                'discipline_score': 0,
                'avg_payment_gap': 0,
                'consistency_rating': 'Error',
                'frequency_pattern': 'Error',
                'recent_trend': 'Error',
                'overdue_count': 0,
                'overdue_amount': 0,
                'longest_overdue_days': 0,
                'risk_level': 'Unknown',
                'reliability_rating': 'Unknown',
                'next_due_date': None,
                'next_due_amount': 0,
                'monthly_obligation': 0,
                'last_payment_date': None,
                'days_since_last_payment': None
            }
    
    def generate_text_summary(self, data: Dict[str, Any], payment_analysis: Dict[str, Any]) -> str:
        """
        Generate a concise text summary of borrower's loan portfolio. This must be in the form of a paragraph with clear sections. make it short and crisp but as sentences and not bullet points. do not use bullet points and special characters
        
        Args:
            data: Complete borrower data
            payment_analysis: Detailed payment analysis
            
        Returns:
            Multi-line text summary
        """
        try:
            borrower_id = data['borrower_info']['borrower_id']
            cibil_score = data['borrower_info']['cibil_score']
            account_age = (datetime.now() - datetime.strptime(data['borrower_info']['created_date'], '%Y-%m-%d')).days
            
            # Portfolio metrics
            total_loans = data['portfolio_summary']['total_loans']
            total_principal = data['portfolio_summary']['total_principal']
            total_paid = data['portfolio_summary']['total_paid']
            total_pending = data['portfolio_summary']['total_pending']
            completion_rate = data['portfolio_summary']['payment_completion_rate']
            
            # Payment behavior
            paid_fragments = data['portfolio_summary']['paid_fragments']
            pending_fragments = data['portfolio_summary']['pending_fragments']
            discipline_score = payment_analysis['discipline_score']
            reliability = payment_analysis['reliability_rating']
            risk_level = payment_analysis['risk_level']
            
            # Current status
            next_due_date = payment_analysis.get('next_due_date')
            next_due_amount = payment_analysis.get('next_due_amount', 0)
            monthly_obligation = payment_analysis.get('monthly_obligation', 0)
            overdue_count = payment_analysis.get('overdue_count', 0)
            overdue_amount = payment_analysis.get('overdue_amount', 0)
            
            # Build summary text
            summary_lines = []
            
            # Header
            summary_lines.append(f"=== BORROWER SUMMARY ===")
            summary_lines.append(f"CIBIL Score: {cibil_score} | Account Age: {account_age} days | Risk Level: {risk_level}")
            summary_lines.append("")
            
            # Portfolio Overview
            summary_lines.append("PORTFOLIO OVERVIEW:")
            summary_lines.append(f"• Total Loans: {total_loans} | Principal Amount: ₹{total_principal:,.2f}")
            summary_lines.append(f"• Amount Paid: ₹{total_paid:,.2f} | Pending: ₹{total_pending:,.2f}")
            summary_lines.append(f"• Completion Rate: {completion_rate}% | Fragments: {paid_fragments}/{paid_fragments + pending_fragments}")
            summary_lines.append("")
            
            # Payment Behavior
            summary_lines.append("PAYMENT BEHAVIOR:")
            summary_lines.append(f"• Payment Discipline: {discipline_score}% | Reliability: {reliability}")
            summary_lines.append(f"• On-time: {payment_analysis.get('on_time_count', 0)} | Early: {payment_analysis.get('early_count', 0)} | Late: {payment_analysis.get('late_count', 0)}")
            summary_lines.append(f"• Average Delay: {payment_analysis.get('avg_delay_days', 0)} days | Max Delay: {payment_analysis.get('max_delay_days', 0)} days")
            summary_lines.append(f"• Payment Pattern: {payment_analysis.get('frequency_pattern', 'Unknown')} | Consistency: {payment_analysis.get('consistency_rating', 'Unknown')}")
            summary_lines.append("")
            
            # Current Status
            summary_lines.append("CURRENT STATUS:")
            if overdue_count > 0:
                summary_lines.append(f"• OVERDUE: {overdue_count} installments worth ₹{overdue_amount:,.2f}")
            else:
                summary_lines.append("• No overdue payments")
            
            if next_due_date and next_due_amount > 0:
                summary_lines.append(f"• Next Due: ₹{next_due_amount:,.2f} on {next_due_date}")
            else:
                summary_lines.append("• No upcoming payments")
            
            if monthly_obligation > 0:
                summary_lines.append(f"• Monthly Obligation: ₹{monthly_obligation:,.2f}")
            
            # Recent Activity
            last_payment = payment_analysis.get('last_payment_date')
            days_since_last = payment_analysis.get('days_since_last_payment')

            
            summary_lines.append("")
            
            # Active Loans Status
            active_loans = len([loan for loan in data['loans'] if loan['status'] in ['Funded', 'Requested']])
            completed_loans = len([loan for loan in data['loans'] if loan['status'] == 'Repaid'])
            defaulted_loans = len([loan for loan in data['loans'] if loan['status'] == 'Defaulted'])
            
            summary_lines.append("LOAN STATUS BREAKDOWN:")
            summary_lines.append(f"• Active: {active_loans} | Completed: {completed_loans} | Defaulted: {defaulted_loans}")
            
            # Risk Assessment
            summary_lines.append("")
            summary_lines.append("RISK ASSESSMENT:")
            if risk_level == "High Risk":
                summary_lines.append("• HIGH RISK: Significant payment delays or defaults detected")
            elif risk_level == "Medium Risk":
                summary_lines.append("• MEDIUM RISK: Some payment irregularities observed")
            else:
                summary_lines.append("• LOW RISK: Good payment history and compliance")
            
            # Trend Analysis
            trend = payment_analysis.get('recent_trend', 'Unknown')
            summary_lines.append(f"• Recent Trend: {trend}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_borrower(self, borrower_id: str) -> Dict[str, Any]:
        """
        Main API method to analyze borrower
        
        Args:
            borrower_id: Unique identifier for the borrower
            
        Returns:
            Complete analysis response
        """
        # Get complete data
        data = self.get_borrower_complete_data(borrower_id)
        
        if "error" in data:
            return data
        
        # Generate AI analysis
        ai_analysis = self.analyze_loan_performance(data)
        
        # Create response
        response = {
            "borrower_id": borrower_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "borrower_info": data['borrower_info'],
            "portfolio_summary": data['portfolio_summary'],
            "ai_analysis": ai_analysis,
            "loans": data['loans']
        }
        
        return response


# Initialize FastAPI app
app = FastAPI(
    title="Loan Analysis API",
    description="API for analyzing borrower loan performance and payment patterns",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

MONGODB_URL = "mongodb://deficosmos:CJurC3DSN6s5JjlkEjxDcvznpK9HI77ibKE1mOmy4mPnxRypzv6iKpjNGBQRQFYyUUqbS78D2D2uACDbryiR7Q%3D%3D@deficosmos.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@deficosmos@Meta"
GEMINI_API_KEY = "AIzaSyCq_6Wg1d5CSo3Z5usHII1-LuPeRIPWr44"

loan_api = LoanAnalysisAPI(MONGODB_URL, GEMINI_API_KEY)

@app.get("/analyze/{borrower_id}")
async def analyze_borrower_endpoint(borrower_id: str):
    """
    API endpoint to analyze borrower
    
    Args:
        borrower_id: Borrower ID from URL path
        
    Returns:
        JSON response with analysis
    """
    try:
        result = loan_api.analyze_borrower(borrower_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "loan-analysis-api"}

@app.get("/borrower/{borrower_id}/summary", response_class=PlainTextResponse)
async def get_borrower_summary(borrower_id: str):
    """
    Get detailed borrower summary as formatted text
    
    Args:
        borrower_id: Borrower ID from URL path
        
    Returns:
        Plain text response with detailed summary
    """
    try:
        data = loan_api.get_borrower_complete_data(borrower_id)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        
        # Calculate detailed payment metrics
        payment_analysis = loan_api.calculate_detailed_payment_summary(data)
        
        # Generate text summary
        text_summary = loan_api.generate_text_summary(data, payment_analysis)
        return PlainTextResponse(text_summary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("agen:app", host="0.0.0.0", port=5000, reload=True)