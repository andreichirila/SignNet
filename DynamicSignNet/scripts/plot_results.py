import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results():
    # 1. Daten laden
    try:
        df_train_loss = pd.read_csv('train_loss.csv')
        df_val_loss = pd.read_csv('val_loss.csv')
        df_train_acc = pd.read_csv('train_accuracy.csv')
        df_val_acc = pd.read_csv('val_accuracy.csv')
        df_lr = pd.read_csv('learning_rate.csv') # NEU: Lernraten-Daten
    except FileNotFoundError as e:
        print(f"Fehler: Eine der CSV-Dateien wurde nicht gefunden: {e}")
        return

    # Daten sortieren
    dfs = [df_train_loss, df_val_loss, df_train_acc, df_val_acc, df_lr]
    for df in dfs:
        df.sort_values('step', inplace=True)

    # 2. Plot-Styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 3. Grafik für Loss mit zweiter Achse für Learning Rate
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Linke Achse: Loss
    lns1 = ax1.plot(df_train_loss['step'], df_train_loss['value'], label='Training Loss', color='#1f77b4', linewidth=1.5, alpha=0.8)
    lns2 = ax1.plot(df_val_loss['step'], df_val_loss['value'], label='Validation Loss', color='#ff7f0e', linewidth=2)
    ax1.set_xlabel('Trainingsschritte (Steps)', fontsize=12)
    ax1.set_ylabel('Balanced Softmax Loss', fontsize=12)
    ax1.set_title('Loss-Verlauf und Learning Rate Restarts (Phase 9)', fontsize=14, pad=15)

    # Rechte Achse: Learning Rate
    ax2 = ax1.twinx()
    # Farbe auf Lila geändert, Linienstärke erhöht, Alpha auf 1.0 (voll deckend)
    lns3 = ax2.plot(df_lr['step'], df_lr['value'], label='Learning Rate', 
                    color='#9467bd', linestyle='--', linewidth=1.5, alpha=1.0)
    
    ax2.set_yscale('log') # Logarithmisch wegen 1e-4 bis 1e-7
    ax2.set_ylabel('Learning Rate (log-scale)', fontsize=12, color='#9467bd', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#9467bd')

    # Kombinierte Legende
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', fontsize=10, frameon=True, shadow=True)

    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loss_plot_with_lr.png', dpi=300)
    print("Grafik gespeichert: loss_plot_with_lr.png")
    plt.show()

    # 4. Grafik für Accuracy (unverändert, aber im selben Stil)
    plt.figure(figsize=(12, 7))
    plt.plot(df_train_acc['step'], df_train_acc['value'], label='Training Accuracy', color='#2ca02c', linewidth=1.5, alpha=0.8)
    plt.plot(df_val_acc['step'], df_val_acc['value'], label='Validation Accuracy', color='#d62728', linewidth=2)
    
    plt.title('Training und Validation Accuracy (SignNet Phase 9)', fontsize=14, pad=15)
    plt.xlabel('Trainingsschritte (Steps)', fontsize=12)
    plt.ylabel('Accuracy (0.0 - 1.0)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_plot_final.png', dpi=300)
    print("Grafik gespeichert: accuracy_plot_final.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()