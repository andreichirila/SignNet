import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results():
    # 1. Daten laden
    # Die Dateien müssen im selben Ordner wie das Skript liegen
    try:
        df_train_loss = pd.read_csv('train_loss.csv')
        df_val_loss = pd.read_csv('val_loss.csv')
        df_train_acc = pd.read_csv('train_accuracy.csv')
        df_val_acc = pd.read_csv('val_accuracy.csv')
    except FileNotFoundError as e:
        print(f"Fehler: Eine der CSV-Dateien wurde nicht gefunden: {e}")
        return

    # Daten sortieren (falls die Schritte nicht in Reihenfolge sind)
    df_train_loss = df_train_loss.sort_values('step')
    df_val_loss = df_val_loss.sort_values('step')
    df_train_acc = df_train_acc.sort_values('step')
    df_val_acc = df_val_acc.sort_values('step')

    # 2. Plot-Styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 3. Grafik für Loss (Verlust)
    plt.figure(figsize=(10, 6))
    plt.plot(df_train_loss['step'], df_train_loss['value'], label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(df_val_loss['step'], df_val_loss['value'], label='Validation Loss', color='#ff7f0e', linewidth=2)
    
    plt.title('Training und Validation Loss (SignNet Phase 9)', fontsize=14, pad=15)
    plt.xlabel('Trainingsschritte (Steps)', fontsize=12)
    plt.ylabel('Loss-Wert', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loss_plot_final.png', dpi=300)
    print("Grafik gespeichert: loss_plot_final.png")
    plt.show()

    # 4. Grafik für Accuracy (Genauigkeit)
    plt.figure(figsize=(10, 6))
    plt.plot(df_train_acc['step'], df_train_acc['value'], label='Training Accuracy', color='#2ca02c', linewidth=2)
    plt.plot(df_val_acc['step'], df_val_acc['value'], label='Validation Accuracy', color='#d62728', linewidth=2)
    
    plt.title('Training und Validation Accuracy (SignNet Phase 9)', fontsize=14, pad=15)
    plt.xlabel('Trainingsschritte (Steps)', fontsize=12)
    plt.ylabel('Accuracy (0.0 - 1.0)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_plot_final.png', dpi=300)
    print("Grafik gespeichert: accuracy_plot_final.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()