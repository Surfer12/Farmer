#!/usr/bin/env python3
"""
Interactive PyTorch Learning Script
Run this daily to practice PyTorch concepts with your Farmer project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from python.enhanced_psi_minimal import EnhancedPsiFramework

def lesson_1_tensors():
    """Lesson 1: Master tensor operations"""
    print("🔥 LESSON 1: Tensor Operations")
    print("-" * 40)
    
    # Create different types of tensors
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.randn(3)
    
    print(f"Tensor a: {a}")
    print(f"Random tensor b: {b}")
    print(f"Addition: {a + b}")
    print(f"Element-wise multiplication: {a * b}")
    print(f"Dot product: {torch.dot(a, b):.4f}")
    
    # Matrix operations
    matrix = torch.randn(3, 3)
    result = torch.mm(matrix, a.unsqueeze(1))  # Matrix-vector multiplication
    print(f"Matrix-vector result shape: {result.shape}")
    
    return a, b, matrix

def lesson_2_gradients():
    """Lesson 2: Understand automatic differentiation"""
    print("\n🔥 LESSON 2: Automatic Differentiation")
    print("-" * 40)
    
    # Simple function with gradient
    x = torch.tensor(3.0, requires_grad=True)
    y = 2 * x**3 + x**2 - 5*x + 1
    
    print(f"x = {x.item()}")
    print(f"y = 2x³ + x² - 5x + 1 = {y.item():.4f}")
    
    # Compute gradient
    y.backward()
    print(f"dy/dx = 6x² + 2x - 5 = {x.grad.item():.4f}")
    print(f"Expected at x=3: 6(9) + 2(3) - 5 = {6*9 + 6 - 5}")
    
    return x, y

def lesson_3_neural_network():
    """Lesson 3: Build and train a neural network"""
    print("\n🔥 LESSON 3: Neural Network Training")
    print("-" * 40)
    
    # Define a simple network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x
    
    # Create model and data
    model = SimpleNet()
    X = torch.randn(10, 2)  # 10 samples, 2 features
    y = torch.randint(0, 2, (10, 1)).float()  # Binary targets
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training for 50 epochs...")
    for epoch in range(50):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model, X, y

def lesson_4_farmer_integration():
    """Lesson 4: Apply PyTorch to your Farmer project"""
    print("\n🔥 LESSON 4: Farmer-PyTorch Integration")
    print("-" * 40)
    
    # Initialize Farmer framework
    framework = EnhancedPsiFramework()
    
    # Test different types of content
    contents = [
        "Simple mathematical equation: x + y = z",
        "Complex differential equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²",
        "Neural network architecture with backpropagation",
        "Physics-informed neural networks for PDEs",
        "Hybrid symbolic-neural reasoning systems"
    ]
    
    # Collect results
    results = []
    for i, content in enumerate(contents):
        result = framework.compute_enhanced_psi(content, 'md', t=float(i))
        results.append([
            len(content) / 100.0,  # Content length feature
            result['S_x'],         # Symbolic accuracy
            result['N_x'],         # Neural accuracy
            result['psi_final']    # Target Ψ(x)
        ])
    
    # Convert to tensors
    data_tensor = torch.tensor(results)
    features = data_tensor[:, :3]  # First 3 columns as features
    targets = data_tensor[:, 3:4]  # Last column as target
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample features: {features[0]}")
    print(f"Sample target: {targets[0].item():.4f}")
    
    # Simple correlation analysis
    correlation = torch.corrcoef(data_tensor.T)
    print(f"Feature-target correlations:")
    print(f"Length-Ψ(x): {correlation[0, 3].item():.3f}")
    print(f"Symbolic-Ψ(x): {correlation[1, 3].item():.3f}")
    print(f"Neural-Ψ(x): {correlation[2, 3].item():.3f}")
    
    return features, targets, correlation

def main():
    """Run all lessons"""
    print("🚀 PyTorch Learning Session")
    print("=" * 50)
    
    # Run lessons
    lesson_1_tensors()
    lesson_2_gradients()
    lesson_3_neural_network()
    lesson_4_farmer_integration()
    
    print("\n🎉 Great job! You've completed today's PyTorch learning session.")
    print("\nKey concepts you practiced:")
    print("✅ Tensor creation and operations")
    print("✅ Automatic differentiation")
    print("✅ Neural network training")
    print("✅ Integration with your Farmer project")
    
    print("\n📚 Tomorrow's topics:")
    print("• Custom loss functions")
    print("• Model saving and loading")
    print("• Advanced optimizers")
    print("• Physics-informed neural networks")

if __name__ == "__main__":
    main()
