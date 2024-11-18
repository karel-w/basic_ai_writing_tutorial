# AI Tutorial Installation Guide

## Python Virtual Environment Setup

1. Create a virtual environment:
```bash
python3.9 -m venv AI_tutorial
```

2. Activate the virtual environment:
```bash
source AI_tutorial/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Deactivate the environment (when done):
```bash
deactivate
```

## Additional Installations

5. Install Jupyter:
```bash
pip install jupyter
```

6. Install IPython kernel:
```bash
python -m ipykernel install --user --name AI_tutorial
```

## Important Considerations

### GPU and PyTorch Installation

- **Check GPU and CUDA Version**: 
  - Verify your GPU and CUDA compatibility
- **PyTorch Installation**: 
  - Visit [PyTorch Official Website](https://pytorch.org/get-started/locally/)
  - Select the appropriate version for your system

### Optional: VSCode and SSH Setup

- Configure VSCode remote SSH connection to local machine
- Ensure proper SSH key setup and configuration

**Note**: Always refer to the most recent documentation for the most up-to-date installation instructions.