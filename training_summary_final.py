import os
import pandas as pd
import json

def show_training_summary():
    """Hiển thị tóm tắt tất cả các cải tiến và kết quả"""
    
    print("🚀 TRAFFIC SIGN DETECTION - FINAL TRAINING SUMMARY")
    print("=" * 70)
    
    # Kiểm tra các session training
    training_history_dir = "training_history"
    if not os.path.exists(training_history_dir):
        print("❌ No training history found!")
        return
    
    sessions = [d for d in os.listdir(training_history_dir) if d.startswith('train')]
    if not sessions:
        print("❌ No training sessions found!")
        return
    
    # Sắp xếp theo số session
    sessions_sorted = sorted(sessions, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
    
    print(f"📊 Found {len(sessions_sorted)} training sessions:")
    for session in sessions_sorted:
        print(f"   - {session}")
    
    print("\n" + "=" * 70)
    
    # Hiển thị kết quả từng session
    for session in sessions_sorted:
        session_path = os.path.join(training_history_dir, session)
        display_session_summary(session_path)
        print("\n" + "-" * 50)

def display_session_summary(session_dir):
    """Hiển thị tóm tắt cho một session"""
    
    session_name = os.path.basename(session_dir)
    print(f"\n📊 SESSION: {session_name}")
    
    # Load configuration
    config_file = os.path.join(session_dir, "configs", "training_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"🔧 Configuration:")
        print(f"   Model: {config['model_config']['model_size']}")
        print(f"   Epochs: {config['model_config']['epochs']}")
        print(f"   Batch Size: {config['model_config']['batch_size']}")
        print(f"   Image Size: {config['model_config']['image_size']}")
        print(f"   Learning Rate: {config['model_config']['learning_rate']}")
        print(f"   Start Fresh: {'Yes' if config['session_info'].get('continue_from') is None else 'No'}")
    
    # Load results
    results_file = os.path.join(session_dir, "results", "training_results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        
        if len(df) > 0:
            last_epoch = df.iloc[-1]
            
            print(f"\n🏆 Training Results (Epoch {last_epoch['epoch']}):")
            print(f"   ⏱️  Training Time: {last_epoch['time']:.2f} seconds")
            print(f"   📈 mAP50: {last_epoch['metrics/mAP50(B)']:.4f}")
            print(f"   📊 mAP50-95: {last_epoch['metrics/mAP50-95(B)']:.4f}")
            print(f"   🎯 Precision: {last_epoch['metrics/precision(B)']:.4f}")
            print(f"   🔍 Recall: {last_epoch['metrics/recall(B)']:.4f}")
            
            # Performance assessment
            print(f"\n📋 Performance Assessment:")
            map50 = last_epoch['metrics/mAP50(B)']
            precision = last_epoch['metrics/precision(B)']
            recall = last_epoch['metrics/recall(B)']
            
            if map50 > 0.7:
                print("   ✅ Excellent mAP50 (>0.7)")
            elif map50 > 0.5:
                print("   ✅ Good mAP50 (>0.5)")
            else:
                print("   ⚠️  mAP50 needs improvement (<0.5)")
            
            if precision > 0.7:
                print("   ✅ Excellent Precision (>0.7)")
            elif precision > 0.6:
                print("   ✅ Good Precision (>0.6)")
            else:
                print("   ⚠️  Precision needs improvement (<0.6)")
            
            if recall > 0.6:
                print("   ✅ Good Recall (>0.6)")
            elif recall > 0.5:
                print("   ✅ Acceptable Recall (>0.5)")
            else:
                print("   ⚠️  Recall needs improvement (<0.5)")
    
    # Check for generated plots
    plots_dir = os.path.join(session_dir, "plots")
    if os.path.exists(plots_dir):
        plot_files = os.listdir(plots_dir)
        if plot_files:
            print(f"\n📊 Generated Plots ({len(plot_files)} files):")
            for plot_file in plot_files:
                file_path = os.path.join(plots_dir, plot_file)
                if plot_file.endswith('.png'):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   📈 {plot_file} ({size_mb:.1f} MB)")
                elif plot_file.endswith('.csv'):
                    print(f"   📋 {plot_file}")
    
    # Check for model weights
    weights_dir = os.path.join(session_dir, "weights")
    if os.path.exists(weights_dir):
        weight_files = os.listdir(weights_dir)
        if weight_files:
            print(f"\n💾 Model Weights:")
            for weight_file in weight_files:
                if weight_file.endswith('.pt'):
                    weight_path = os.path.join(weights_dir, weight_file)
                    size_mb = os.path.getsize(weight_path) / (1024 * 1024)
                    print(f"   🎯 {weight_file} ({size_mb:.1f} MB)")

def show_improvements():
    """Hiển thị các cải tiến đã thực hiện"""
    
    print("\n" + "=" * 70)
    print("🔧 IMPROVEMENTS IMPLEMENTED")
    print("=" * 70)
    
    improvements = [
        "✅ Fixed dataset path from 'datasetv2' to 'dataset'",
        "✅ Fixed multiprocessing issue (workers=0 for Windows)",
        "✅ Made each training session completely independent",
        "✅ Added comprehensive plot generation",
        "✅ Fixed predict2.py model loading path",
        "✅ Added detailed training curves",
        "✅ Added performance metrics analysis",
        "✅ Added loss analysis",
        "✅ Added learning rate analysis",
        "✅ Added training summary reports",
        "✅ Added overfitting detection",
        "✅ Added efficiency analysis",
        "✅ Improved error handling and logging"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print("\n📊 PLOTS GENERATED:")
    plot_types = [
        "📈 Training Curves (6 subplots)",
        "📊 Performance Metrics Comparison",
        "📉 Loss Analysis (Training vs Validation)",
        "📋 Training Summary CSV",
        "🎯 Learning Rate Schedule",
        "📈 Confusion Matrix",
        "📊 Precision-Recall Curves",
        "📈 F1 Score Curves"
    ]
    
    for plot_type in plot_types:
        print(f"   {plot_type}")

def show_usage_instructions():
    """Hiển thị hướng dẫn sử dụng"""
    
    print("\n" + "=" * 70)
    print("📖 USAGE INSTRUCTIONS")
    print("=" * 70)
    
    instructions = [
        "🚀 To start a new training session:",
        "   python train.py",
        "",
        "📊 To view training results:",
        "   python view_training_results.py",
        "",
        "🎯 To run real-time detection:",
        "   python predict2.py",
        "",
        "📈 To generate plots for any session:",
        "   python generate_plots.py",
        "",
        "📁 Plot locations:",
        "   training_history/trainX/plots/",
        "",
        "💾 Model weights:",
        "   training_history/trainX/weights/best.pt"
    ]
    
    for instruction in instructions:
        print(f"   {instruction}")

if __name__ == "__main__":
    show_training_summary()
    show_improvements()
    show_usage_instructions()
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING SYSTEM READY!")
    print("📊 All plots and visualizations are now available")
    print("🎯 Each training session is completely independent")
    print("=" * 70) 