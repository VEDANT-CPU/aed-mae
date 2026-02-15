import matplotlib.pyplot as plt
import re

def plot_loss(log_file_path):
    epochs = []
    losses = []
    
    # Regular expressions to find the epoch and the averaged loss
    # We look for "Epoch: [X] Total time" followed by the "Averaged stats" line
    epoch_pattern = re.compile(r"Epoch: \[(\d+)\] Total time")
    loss_pattern = re.compile(r"Averaged stats: .* loss: [\d\.]+ \((\d+\.\d+)\)")

    current_epoch = None
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            # Check for epoch number
            epoch_match = epoch_pattern.search(lines[i])
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                
                # The next line usually contains the "Averaged stats"
                if i + 1 < len(lines):
                    loss_match = loss_pattern.search(lines[i+1])
                    if loss_match:
                        epochs.append(current_epoch)
                        # We extract the value in parentheses (the epoch average)
                        losses.append(float(loss_match.group(1)))

    if not epochs:
        print("No training data found in log file.")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss vs. Epoch (Avenue Dataset)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig('loss_curve.png')
    print(f"Plot saved as loss_curve.png. Processed {len(epochs)} epochs.")

# Run the script
plot_loss('training_log.txt')