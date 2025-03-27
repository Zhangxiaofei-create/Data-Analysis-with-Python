# Load the model
try:
    # Option 1: Load state_dict
    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load("model_state_dict.pth"))
    loaded_model.eval()
    
    # Option 2: Load scripted model
    loaded_scripted_model = torch.jit.load("scripted_model.pth")
    loaded_scripted_model.eval()
    
    # Test the loaded model
    x = torch.randn((1, 3, 32, 32))
    out = loaded_model(x)
    print("Loaded model output shape:", out.shape)
    
    out_scripted = loaded_scripted_model(x)
    print("Scripted model output shape:", out_scripted.shape)
    
except Exception as e:
    print(f"Error loading model: {e}")
