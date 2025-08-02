import os
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class TrainingManager:
    """Quản lý lịch sử training"""
    
    def __init__(self):
        self.training_history_dir = 'training_history'
        self.training_log_file = os.path.join(self.training_history_dir, 'training_log.json')
        self.load_training_log()
    
    def load_training_log(self):
        """Load training log"""
        if os.path.exists(self.training_log_file):
            with open(self.training_log_file, 'r', encoding='utf-8') as f:
                self.training_log = json.load(f)
        else:
            self.training_log = {
                'total_training_sessions': 0,
                'sessions': []
            }
    
    def list_sessions(self, detailed=False):
        """Liệt kê tất cả các session training"""
        print("\n" + "="*60)
        print("TRAINING SESSIONS HISTORY")
        print("="*60)
        
        if not self.training_log['sessions']:
            print("❌ No training sessions found.")
            return
        
        print(f"📊 Total sessions: {self.training_log['total_training_sessions']}")
        print(f"📁 History directory: {self.training_history_dir}")
        print("-" * 60)
        
        for session in self.training_log['sessions']:
            self.print_session_info(session, detailed)
            print("-" * 60)
    
    def print_session_info(self, session, detailed=False):
        """In thông tin chi tiết của một session"""
        session_num = session['session_number']
        status = session['status']
        
        # Icon cho status
        status_icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "⏳"
        
        print(f"\n{status_icon} Session {session_num}")
        print(f"   📂 Directory: {session['session_dir']}")
        print(f"   🕐 Start Time: {session['start_time']}")
        print(f"   📊 Status: {status}")
        
        if session.get('continue_from'):
            print(f"   🔄 Continued from: Session {session['continue_from']}")
        
        if status == 'completed':
            print(f"   🎯 Best mAP50: {session.get('best_map50', 0):.4f}")
            print(f"   🎯 Best mAP50-95: {session.get('best_map50_95', 0):.4f}")
            print(f"   📈 Final Epoch: {session.get('final_epoch', 0)}")
            if session.get('end_time'):
                print(f"   🕐 End Time: {session['end_time']}")
        elif status == 'failed':
            print(f"   ❌ Error: {session.get('error', 'Unknown error')}")
        
        if detailed:
            self.print_session_details(session)
    
    def print_session_details(self, session):
        """In chi tiết về files và cấu trúc thư mục của session"""
        session_dir = session['session_dir']
        if not os.path.exists(session_dir):
            print("   ⚠️  Session directory not found!")
            return
        
        print("\n   📁 Directory Structure:")
        for root, dirs, files in os.walk(session_dir):
            level = root.replace(session_dir, '').count(os.sep)
            indent = '   ' * (level + 1)
            folder_name = os.path.basename(root)
            if folder_name:
                print(f"{indent}📁 {folder_name}/")
            
            for file in files[:5]:  # Chỉ hiển thị 5 file đầu
                print(f"{indent}   📄 {file}")
            if len(files) > 5:
                print(f"{indent}   ... and {len(files) - 5} more files")
    
    def get_best_session(self):
        """Tìm session có kết quả tốt nhất"""
        completed_sessions = [s for s in self.training_log['sessions'] if s['status'] == 'completed']
        
        if not completed_sessions:
            print("❌ No completed sessions found.")
            return None
        
        # Sắp xếp theo mAP50
        best_session = max(completed_sessions, key=lambda x: x.get('best_map50', 0))
        
        print(f"\n🏆 Best Session: {best_session['session_number']}")
        print(f"   📊 mAP50: {best_session.get('best_map50', 0):.4f}")
        print(f"   📊 mAP50-95: {best_session.get('best_map50_95', 0):.4f}")
        print(f"   📂 Directory: {best_session['session_dir']}")
        
        return best_session
    
    def copy_session(self, session_num, new_name=None):
        """Copy một session để tạo bản sao"""
        session = self.get_session_by_number(session_num)
        if not session:
            print(f"❌ Session {session_num} not found.")
            return
        
        source_dir = session['session_dir']
        if not os.path.exists(source_dir):
            print(f"❌ Session directory not found: {source_dir}")
            return
        
        if new_name is None:
            new_name = f"session_{session_num}_copy"
        
        dest_dir = os.path.join(self.training_history_dir, new_name)
        
        try:
            shutil.copytree(source_dir, dest_dir)
            print(f"✅ Copied session {session_num} to: {dest_dir}")
        except Exception as e:
            print(f"❌ Error copying session: {e}")
    
    def get_session_by_number(self, session_num):
        """Tìm session theo số"""
        for session in self.training_log['sessions']:
            if session['session_number'] == session_num:
                return session
        return None
    
    def delete_session(self, session_num):
        """Xóa một session"""
        session = self.get_session_by_number(session_num)
        if not session:
            print(f"❌ Session {session_num} not found.")
            return
        
        session_dir = session['session_dir']
        if not os.path.exists(session_dir):
            print(f"❌ Session directory not found: {session_dir}")
            return
        
        try:
            shutil.rmtree(session_dir)
            # Xóa khỏi training log
            self.training_log['sessions'] = [s for s in self.training_log['sessions'] if s['session_number'] != session_num]
            self.save_training_log()
            print(f"✅ Deleted session {session_num}")
        except Exception as e:
            print(f"❌ Error deleting session: {e}")
    
    def save_training_log(self):
        """Lưu training log"""
        with open(self.training_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
    
    def create_summary_report(self):
        """Tạo báo cáo tổng hợp"""
        if not self.training_log['sessions']:
            print("❌ No sessions to report.")
            return
        
        completed_sessions = [s for s in self.training_log['sessions'] if s['status'] == 'completed']
        failed_sessions = [s for s in self.training_log['sessions'] if s['status'] == 'failed']
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY REPORT")
        print("="*60)
        
        print(f"📊 Total Sessions: {len(self.training_log['sessions'])}")
        print(f"✅ Completed: {len(completed_sessions)}")
        print(f"❌ Failed: {len(failed_sessions)}")
        
        if completed_sessions:
            best_map50 = max([s.get('best_map50', 0) for s in completed_sessions])
            best_map50_95 = max([s.get('best_map50_95', 0) for s in completed_sessions])
            avg_map50 = sum([s.get('best_map50', 0) for s in completed_sessions]) / len(completed_sessions)
            
            print(f"\n🏆 Best Performance:")
            print(f"   📊 Best mAP50: {best_map50:.4f}")
            print(f"   📊 Best mAP50-95: {best_map50_95:.4f}")
            print(f"   📊 Average mAP50: {avg_map50:.4f}")
        
        # Tạo biểu đồ nếu có matplotlib
        try:
            self.create_performance_chart()
        except Exception as e:
            print(f"⚠️  Could not create performance chart: {e}")
    
    def create_performance_chart(self):
        """Tạo biểu đồ hiệu suất"""
        completed_sessions = [s for s in self.training_log['sessions'] if s['status'] == 'completed']
        
        if len(completed_sessions) < 2:
            print("📊 Need at least 2 completed sessions to create chart.")
            return
        
        session_numbers = [s['session_number'] for s in completed_sessions]
        map50_scores = [s.get('best_map50', 0) for s in completed_sessions]
        map50_95_scores = [s.get('best_map50_95', 0) for s in completed_sessions]
        
        plt.figure(figsize=(12, 6))
        plt.plot(session_numbers, map50_scores, 'o-', label='mAP50', linewidth=2, markersize=8)
        plt.plot(session_numbers, map50_95_scores, 's-', label='mAP50-95', linewidth=2, markersize=8)
        
        plt.xlabel('Training Session')
        plt.ylabel('mAP Score')
        plt.title('Training Performance Over Sessions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Lưu biểu đồ
        chart_path = os.path.join(self.training_history_dir, 'performance_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance chart saved to: {chart_path}")
    
    def export_to_csv(self):
        """Xuất dữ liệu training ra CSV"""
        if not self.training_log['sessions']:
            print("❌ No sessions to export.")
            return
        
        data = []
        for session in self.training_log['sessions']:
            row = {
                'session_number': session['session_number'],
                'status': session['status'],
                'start_time': session['start_time'],
                'end_time': session.get('end_time', ''),
                'best_map50': session.get('best_map50', 0),
                'best_map50_95': session.get('best_map50_95', 0),
                'final_epoch': session.get('final_epoch', 0),
                'continue_from': session.get('continue_from', ''),
                'session_dir': session['session_dir']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.training_history_dir, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Training history exported to: {csv_path}")
        return csv_path

def main():
    """Main function"""
    manager = TrainingManager()
    
    print("🚀 Training Manager")
    print("Available commands:")
    print("1. list - List all sessions")
    print("2. list -d - List with details")
    print("3. best - Show best session")
    print("4. summary - Create summary report")
    print("5. export - Export to CSV")
    print("6. copy <session_num> - Copy session")
    print("7. delete <session_num> - Delete session")
    
    while True:
        try:
            command = input("\nEnter command (or 'quit' to exit): ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
            elif command == 'list':
                manager.list_sessions()
            elif command == 'list -d':
                manager.list_sessions(detailed=True)
            elif command == 'best':
                manager.get_best_session()
            elif command == 'summary':
                manager.create_summary_report()
            elif command == 'export':
                manager.export_to_csv()
            elif command.startswith('copy '):
                try:
                    session_num = int(command.split()[1])
                    manager.copy_session(session_num)
                except (IndexError, ValueError):
                    print("❌ Usage: copy <session_number>")
            elif command.startswith('delete '):
                try:
                    session_num = int(command.split()[1])
                    confirm = input(f"Are you sure you want to delete session {session_num}? (y/N): ")
                    if confirm.lower() == 'y':
                        manager.delete_session(session_num)
                except (IndexError, ValueError):
                    print("❌ Usage: delete <session_number>")
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 