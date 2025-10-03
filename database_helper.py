"""
Database Helper Module
Handles database operations for user feedback, alerts, and case study submissions
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class DatabaseHelper:
    """Helper class for database operations"""
    
    def __init__(self):
        """Initialize database connection"""
        self.connection_string = os.environ.get('DATABASE_URL')
        
    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(self.connection_string)
    
    # ===== User Feedback Operations =====
    
    def submit_feedback(self, name: str, email: str, organization: str, 
                       waterbody_name: str, feedback_type: str, 
                       feedback_text: str, rating: int = None,
                       location_lat: float = None, location_lon: float = None) -> int:
        """
        Submit user feedback
        
        Args:
            name: User's name
            email: User's email
            organization: User's organization
            waterbody_name: Name of waterbody
            feedback_type: Type of feedback ('case_study', 'feedback', 'issue_report')
            feedback_text: Feedback content
            rating: Rating (1-5)
            location_lat: Latitude
            location_lon: Longitude
            
        Returns:
            ID of submitted feedback
        """
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO user_feedback 
                (name, email, organization, waterbody_name, feedback_type, 
                 feedback_text, rating, location_lat, location_lon)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (name, email, organization, waterbody_name, feedback_type,
                  feedback_text, rating, location_lat, location_lon))
            
            feedback_id = cursor.fetchone()[0]
            conn.commit()
            
            return feedback_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def get_all_feedback(self, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all feedback entries
        
        Args:
            status: Filter by status ('pending', 'reviewed', 'approved')
            limit: Maximum number of entries to return
            
        Returns:
            List of feedback entries
        """
        
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if status:
                cursor.execute("""
                    SELECT * FROM user_feedback 
                    WHERE status = %s 
                    ORDER BY submitted_at DESC 
                    LIMIT %s
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM user_feedback 
                    ORDER BY submitted_at DESC 
                    LIMIT %s
                """, (limit,))
            
            feedback_list = cursor.fetchall()
            return [dict(row) for row in feedback_list]
            
        finally:
            cursor.close()
            conn.close()
    
    def update_feedback_status(self, feedback_id: int, status: str, reviewer_notes: str = None):
        """Update feedback status"""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE user_feedback 
                SET status = %s, reviewed_at = CURRENT_TIMESTAMP, reviewer_notes = %s
                WHERE id = %s
            """, (status, reviewer_notes, feedback_id))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    # ===== Alert Subscription Operations =====
    
    def subscribe_to_alerts(self, email: str, name: str, waterbodies: List[str],
                           alert_threshold: str = 'Medium',
                           notification_frequency: str = 'immediate') -> int:
        """
        Subscribe user to algae bloom alerts
        
        Args:
            email: User's email
            name: User's name
            waterbodies: List of waterbody names to monitor
            alert_threshold: Threshold for alerts ('Low', 'Medium', 'High')
            notification_frequency: How often to send alerts
            
        Returns:
            Subscription ID
        """
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Generate verification token
            import secrets
            verification_token = secrets.token_urlsafe(32)
            
            cursor.execute("""
                INSERT INTO alert_subscriptions 
                (email, name, waterbodies, alert_threshold, notification_frequency, verification_token)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (email) 
                DO UPDATE SET 
                    waterbodies = EXCLUDED.waterbodies,
                    alert_threshold = EXCLUDED.alert_threshold,
                    notification_frequency = EXCLUDED.notification_frequency,
                    is_active = TRUE
                RETURNING id
            """, (email, name, waterbodies, alert_threshold, notification_frequency, verification_token))
            
            subscription_id = cursor.fetchone()[0]
            conn.commit()
            
            return subscription_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def get_active_subscriptions(self, waterbody_name: str = None) -> List[Dict[str, Any]]:
        """
        Get active alert subscriptions
        
        Args:
            waterbody_name: Optional filter for specific waterbody
            
        Returns:
            List of active subscriptions
        """
        
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if waterbody_name:
                cursor.execute("""
                    SELECT * FROM alert_subscriptions 
                    WHERE is_active = TRUE 
                    AND verified = TRUE
                    AND %s = ANY(waterbodies)
                    ORDER BY subscribed_at DESC
                """, (waterbody_name,))
            else:
                cursor.execute("""
                    SELECT * FROM alert_subscriptions 
                    WHERE is_active = TRUE AND verified = TRUE
                    ORDER BY subscribed_at DESC
                """)
            
            subscriptions = cursor.fetchall()
            return [dict(row) for row in subscriptions]
            
        finally:
            cursor.close()
            conn.close()
    
    def unsubscribe_from_alerts(self, email: str):
        """Unsubscribe user from alerts"""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE alert_subscriptions 
                SET is_active = FALSE 
                WHERE email = %s
            """, (email,))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def update_last_notification(self, subscription_id: int):
        """Update last notification timestamp"""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE alert_subscriptions 
                SET last_notification_sent = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (subscription_id,))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    # ===== Case Study Submission Operations =====
    
    def submit_case_study(self, submitter_name: str, submitter_email: str,
                         submitter_role: str, waterbody_name: str,
                         observation_date: str, algae_severity: str,
                         estimated_coverage: float = None,
                         location_lat: float = None, location_lon: float = None,
                         observations: str = None, mitigation_attempted: str = None,
                         outcomes: str = None) -> int:
        """
        Submit a case study
        
        Args:
            submitter_name: Name of submitter
            submitter_email: Email of submitter
            submitter_role: Role ('engineer', 'student', 'citizen', 'researcher')
            waterbody_name: Name of waterbody
            observation_date: Date of observation
            algae_severity: Severity level
            estimated_coverage: Estimated coverage percentage
            location_lat: Latitude
            location_lon: Longitude
            observations: Detailed observations
            mitigation_attempted: Mitigation measures attempted
            outcomes: Outcomes of mitigation
            
        Returns:
            Case study ID
        """
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO case_study_submissions 
                (submitter_name, submitter_email, submitter_role, waterbody_name,
                 observation_date, algae_severity, estimated_coverage,
                 location_lat, location_lon, observations, mitigation_attempted, outcomes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (submitter_name, submitter_email, submitter_role, waterbody_name,
                  observation_date, algae_severity, estimated_coverage,
                  location_lat, location_lon, observations, mitigation_attempted, outcomes))
            
            case_study_id = cursor.fetchone()[0]
            conn.commit()
            
            return case_study_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def get_case_studies(self, status: str = None, published_only: bool = False, 
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get case study submissions
        
        Args:
            status: Filter by status
            published_only: Only return published case studies
            limit: Maximum number to return
            
        Returns:
            List of case studies
        """
        
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            query = "SELECT * FROM case_study_submissions WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            if published_only:
                query += " AND published = TRUE"
            
            query += " ORDER BY observation_date DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            
            case_studies = cursor.fetchall()
            return [dict(row) for row in case_studies]
            
        finally:
            cursor.close()
            conn.close()
    
    def approve_case_study(self, case_study_id: int, publish: bool = True):
        """Approve and optionally publish a case study"""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE case_study_submissions 
                SET status = 'approved', 
                    approved_at = CURRENT_TIMESTAMP,
                    published = %s
                WHERE id = %s
            """, (publish, case_study_id))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    # ===== Analysis History Operations =====
    
    def log_analysis(self, waterbody_name: str, analysis_type: str,
                    algae_coverage: float, risk_level: str, risk_score: float,
                    spectral_indices: Dict[str, Any] = None,
                    prediction_data: Dict[str, Any] = None,
                    user_session_id: str = None, ip_address: str = None) -> int:
        """
        Log an analysis to history
        
        Args:
            waterbody_name: Name of waterbody analyzed
            analysis_type: Type of analysis
            algae_coverage: Algae coverage percentage
            risk_level: Risk level
            risk_score: Risk score
            spectral_indices: Spectral indices dict
            prediction_data: ML prediction data dict
            user_session_id: Session ID
            ip_address: User IP address
            
        Returns:
            Analysis ID
        """
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO analysis_history 
                (waterbody_name, analysis_type, algae_coverage, risk_level, risk_score,
                 spectral_indices, prediction_data, user_session_id, ip_address)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (waterbody_name, analysis_type, algae_coverage, risk_level, risk_score,
                  Json(spectral_indices) if spectral_indices else None,
                  Json(prediction_data) if prediction_data else None,
                  user_session_id, ip_address))
            
            analysis_id = cursor.fetchone()[0]
            conn.commit()
            
            return analysis_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def get_analysis_history(self, waterbody_name: str = None, 
                            analysis_type: str = None,
                            days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get analysis history
        
        Args:
            waterbody_name: Filter by waterbody
            analysis_type: Filter by analysis type
            days: Number of days to look back
            limit: Maximum number to return
            
        Returns:
            List of historical analyses
        """
        
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            query = """
                SELECT * FROM analysis_history 
                WHERE analysis_date >= CURRENT_TIMESTAMP - INTERVAL '%s days'
            """
            params = [days]
            
            if waterbody_name:
                query += " AND waterbody_name = %s"
                params.append(waterbody_name)
            
            if analysis_type:
                query += " AND analysis_type = %s"
                params.append(analysis_type)
            
            query += " ORDER BY analysis_date DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            
            history = cursor.fetchall()
            return [dict(row) for row in history]
            
        finally:
            cursor.close()
            conn.close()
    
    def get_waterbody_statistics(self, waterbody_name: str, days: int = 90) -> Dict[str, Any]:
        """
        Get statistics for a specific waterbody
        
        Args:
            waterbody_name: Name of waterbody
            days: Number of days to analyze
            
        Returns:
            Dictionary of statistics
        """
        
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(algae_coverage) as avg_coverage,
                    MAX(algae_coverage) as max_coverage,
                    MIN(algae_coverage) as min_coverage,
                    AVG(risk_score) as avg_risk_score,
                    COUNT(CASE WHEN risk_level = 'High' THEN 1 END) as high_risk_count,
                    COUNT(CASE WHEN risk_level = 'Medium' THEN 1 END) as medium_risk_count,
                    COUNT(CASE WHEN risk_level = 'Low' THEN 1 END) as low_risk_count
                FROM analysis_history
                WHERE waterbody_name = %s
                AND analysis_date >= CURRENT_TIMESTAMP - INTERVAL '%s days'
            """, (waterbody_name, days))
            
            stats = cursor.fetchone()
            return dict(stats) if stats else {}
            
        finally:
            cursor.close()
            conn.close()
