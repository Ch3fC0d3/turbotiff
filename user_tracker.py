
import json
import os
import sqlite3
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any

class UserPreferenceTracker:
    """Phase 1: Basic tracking of user curve adjustments (SQLite backed)"""
    
    def __init__(self, db_path='user_preferences.db'):
        self.db_path = db_path
        self._init_db()
        self.adjustments = defaultdict(list) # Keep in-memory cache structure for compatibility if needed, but rely on DB
    
    def _init_db(self):
        """Initialize the SQLite database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS adjustments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        curve_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        original_params TEXT,
                        user_params TEXT,
                        quality_score REAL,
                        image_context TEXT,
                        session_id TEXT
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")

    def record_adjustment(self, curve_type: str, original_params: Dict, user_params: Dict, quality_score: float = 1.0, image_context: Optional[Dict] = None):
        """Record a user adjustment for learning"""
        timestamp = datetime.now().isoformat()
        session_id = self._get_session_id()
        context = image_context or self._get_image_context()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO adjustments 
                    (curve_type, timestamp, original_params, user_params, quality_score, image_context, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    curve_type,
                    timestamp,
                    json.dumps(original_params),
                    json.dumps(user_params),
                    quality_score,
                    json.dumps(context),
                    session_id
                ))
                conn.commit()
                
                # Maintain in-memory cache for backward compatibility / performance if needed
                # (though strict DB usage is safer)
                self.adjustments[curve_type].append({
                    'timestamp': timestamp,
                    'original_params': original_params,
                    'user_params': user_params,
                    'quality_score': quality_score,
                    'image_context': context,
                    'session_id': session_id
                })
                
        except sqlite3.Error as e:
            print(f"Error recording adjustment: {e}")

    def get_adjustments(self, curve_type: str, limit: Optional[int] = None) -> List[Dict]:
        """Get recent adjustments for a curve type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM adjustments WHERE curve_type = ? ORDER BY timestamp DESC"
                params = [curve_type]
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'timestamp': row['timestamp'],
                        'original_params': json.loads(row['original_params']) if row['original_params'] else {},
                        'user_params': json.loads(row['user_params']) if row['user_params'] else {},
                        'quality_score': row['quality_score'],
                        'image_context': json.loads(row['image_context']) if row['image_context'] else {},
                        'session_id': row['session_id']
                    })
                return results[::-1] # Return in chronological order (oldest first) if that's what caller expects
        except sqlite3.Error as e:
            print(f"Error retrieving adjustments: {e}")
            return []

    def get_all_adjustments(self) -> Dict[str, List[Dict]]:
        """Get all adjustments across all curve types"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT curve_type FROM adjustments")
                curve_types = [row['curve_type'] for row in cursor.fetchall()]
                
                all_adj = {}
                for ct in curve_types:
                    all_adj[ct] = self.get_adjustments(ct)
                return all_adj
        except sqlite3.Error:
            return {}

    def _get_image_context(self):
        """Generate basic context hash for image features"""
        # Simple context - can be enhanced later
        return {
            'width': None,
            'height': None,
            'curve_color': None,
            'noise_level': None
        }
    
    def _get_session_id(self):
        """Generate a simple session identifier"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_stats(self, curve_type: str) -> Dict[str, Any]:
        """Get basic statistics for a curve type"""
        adjustments = self.get_adjustments(curve_type)
        if not adjustments:
            return {'count': 0, 'avg_quality': 0.0}
        
        avg_quality = sum(adj['quality_score'] for adj in adjustments) / len(adjustments)
        return {
            'count': len(adjustments),
            'avg_quality': avg_quality,
            'last_adjustment': adjustments[-1]['timestamp'] if adjustments else None
        }

# Global instance for easy access
tracker = UserPreferenceTracker()
