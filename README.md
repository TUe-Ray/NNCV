# Final Assignment: Cityscape Challenge

Welcome to the repository for **5LSM0: Final Assignment: Cityscape Challenge**,  the final project for the course 5LSM0: Neural Networks for Computer Vision, offered by the Department of Electrical Engineering at Eindhoven University of Technology. This course is hosted by the [Video Coding & Architectures research group](https://www.tue.nl/en/research/research-groups/signal-processing-systems/video-coding-architectures).


## 📦 Required Libraries & Environment Setup

This project uses a pre-configured Docker container with all necessary dependencies:

- **Dataset**: [Cityscapes](https://www.cityscapes-dataset.com/)
- **Docker Image**: `docker://tjmjaspers/nncv2025:v7`
- **Singularity Container**: `container.sif`

### 🔧 Installation Instructions

1. **Clone this repository**:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Download Docker and Cityscapes dataset**:
   Run the following script to download both:
   ```bash
   sbatch download_docker_and_data.sh
   ```

3. **Optional**: To change the Docker version, edit the `download_docker_and_data.sh` script accordingly.

> 💡 All necessary modules are included in the `container.sif` Docker image. No manual pip installations are required.

---


## 🚀 How to Train the Model

1. **Configure the Training Script**:  
   In `main.sh`, replace the script reference with the desired training configuration (e.g., `train_Segformer_Finetuned.py`). All training scripts are prefixed with `train_`.

2. **Adjust Hyperparameters**:  
   Modify learning rate, batch size, and number of epochs directly in `main.sh`.

3. **Set Training Time and GPU Settings**:  
   In `jobscript_slurm.sh`, set the maximum training time and (optionally) specify the number of GPUs.  
   > ⚠️ If you use multiple GPUs, ensure the training script supports parallel training (manual changes may be required).

---

## 🧪 How to Test the Model on Codalab

If you have a trained model checkpoint and want to participate in the Codalab evaluation:

1. **Select Your Model File**:  
   Find the appropriate model script (files prefixed with `Model_`) and rename it to `model.py`.

2. **Update Pre/Post-Processing**:  
   Modify `process_data.py` with the correct input size and any desired preprocessing.

3. **Prepare the Checkpoint**:  
   Rename your `.pth` file to `model.pth`.

4. **Package for Submission**:  
   Zip the following files:
   - `model.py`
   - `model.pth`
   - `process_data.py`

   Submit the zip file to the Codalab competition platform.

---

## 👤 Author & Codalab Info

- **Name**: Shao-Ruei Huang  
- **TU/e Email**: [s.huang5@student.tue.nl](mailto:s.huang5@student.tue.nl)  
- **Codalab Username**: `TUe-Ray`