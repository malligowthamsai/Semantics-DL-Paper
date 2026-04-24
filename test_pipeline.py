import torch
from DSFM_Net import DSFM_Net
from antigravity_net import AntigravityNet, DSFN_Antigravity_Pipeline

def test_pipeline_forward():
    print("Initializing models...")
    dsfn = DSFM_Net(channels=1, classes=1)
    anti_net = AntigravityNet(img_channels=1, mask_channels=1, out_channels=1)
    pipeline = DSFN_Antigravity_Pipeline(dsfn, anti_net)

    # Create dummy input tensor: Batch Size 2, 1 Channel, 512x512
    print("Creating dummy input tensor [2, 1, 512, 512]...")
    dummy_input = torch.randn(2, 1, 512, 512)

    print("Running forward pass...")
    try:
        refined_out, dsfn_out = pipeline(dummy_input)
        
        print("\n--- Pipeline Outputs ---")
        print(f"Refined Output Shape: {refined_out.shape} (Expected: [2, 1, 512, 512])")
        print(f"DSFN Output Shape:    {dsfn_out.shape} (Expected: [2, 1, 512, 512])")
        
        assert refined_out.shape == (2, 1, 512, 512), "Refined output shape mismatch!"
        assert dsfn_out.shape == (2, 1, 512, 512), "DSFN output shape mismatch!"
        
        print("\nSUCCESS: Pipeline forward pass completed with correct shapes.")
    except Exception as e:
        print(f"\nERROR during forward pass: {e}")

if __name__ == '__main__':
    test_pipeline_forward()
