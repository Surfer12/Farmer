#!/usr/bin/env python3
"""
PyTorch Learning Tests - Learn PyTorch while testing Farmer components
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytest
from python.enhanced_psi_minimal import EnhancedPsiFramework

class SimpleNeuralNet(nn.Module):
    """Simple neural network for learning PyTorch basics"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class TestPyTorchBasics:
    """Learn PyTorch fundamentals through testing"""
    
    def test_tensor_creation(self):
        """Learn: Creating and manipulating tensors"""
        # Create tensors different ways
        tensor_from_list = torch.tensor([1.0, 2.0, 3.0])
        tensor_zeros = torch.zeros(3, 4)
        tensor_random = torch.randn(2, 3)
        
        print(f"From list: {tensor_from_list}")
        print(f"Zeros shape: {tensor_zeros.shape}")
        print(f"Random tensor: {tensor_random}")
        
        # Basic operations
        result = tensor_from_list * 2
        assert result.tolist() == [2.0, 4.0, 6.0]
        
    def test_neural_network_creation(self):
        """Learn: Creating and using neural networks"""
        model = SimpleNeuralNet(input_size=5, hidden_size=10, output_size=1)
        
        # Create sample input
        x = torch.randn(1, 5)  # batch_size=1, features=5
        
        # Forward pass
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output value: {output.item():.4f}")
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1  # sigmoid output
    
    def test_training_loop_basics(self):
        """Learn: Basic training loop structure"""
        model = SimpleNeuralNet(input_size=3, hidden_size=5, output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Sample data
        X = torch.randn(10, 3)  # 10 samples, 3 features
        y = torch.randn(10, 1)  # 10 targets
        
        initial_loss = None
        final_loss = None
        
        # Training loop
        for epoch in range(100):
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch == 99:
                final_loss = loss.item()
        
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        
        # Loss should decrease (usually)
        assert final_loss < initial_loss
    
    def test_gradients_and_backprop(self):
        """Learn: Understanding gradients and backpropagation"""
        # Simple function: y = x^2
        x = torch.tensor(2.0, requires_grad=True)
        y = x ** 2
        
        # Compute gradient
        y.backward()
        
        print(f"x = {x.item()}")
        print(f"y = x^2 = {y.item()}")
        print(f"dy/dx = {x.grad.item()}")
        
        # dy/dx should be 2x = 4
        assert abs(x.grad.item() - 4.0) < 1e-6

class TestFarmerPyTorchIntegration:
    """Test PyTorch integration with Farmer components"""
    
    def test_neural_accuracy_with_pytorch(self):
        """Learn: Integrating PyTorch models with Farmer framework"""
        framework = EnhancedPsiFramework()
        
        # Create a simple model for neural accuracy estimation
        model = SimpleNeuralNet(input_size=1, hidden_size=5, output_size=1)
        
        # Test content
        content = "Neural network analysis with PyTorch integration"
        result = framework.compute_enhanced_psi(content, 'md', t=1.0)
        
        # Use PyTorch to enhance neural accuracy
        content_tensor = torch.tensor([[len(content) / 100.0]])  # Simple feature
        neural_enhancement = model(content_tensor).item()
        
        print(f"Original neural accuracy: {result['N_x']:.3f}")
        print(f"PyTorch enhancement: {neural_enhancement:.3f}")
        print(f"Combined accuracy: {(result['N_x'] + neural_enhancement) / 2:.3f}")
        
        assert 0 <= neural_enhancement <= 1
    
    def test_tensor_operations_for_psi(self):
        """Learn: Using tensors for Ψ(x) computations"""
        # Convert Farmer results to tensors for further processing
        framework = EnhancedPsiFramework()
        
        results = []
        for i in range(5):
            content = f"Test content {i} with mathematical structures"
            result = framework.compute_enhanced_psi(content, 'md', t=float(i))
            results.append([
                result['S_x'],
                result['N_x'], 
                result['alpha_t'],
                result['psi_final']
            ])
        
        # Convert to tensor
        results_tensor = torch.tensor(results)
        
        print(f"Results tensor shape: {results_tensor.shape}")
        print(f"Mean Ψ(x): {results_tensor[:, 3].mean().item():.3f}")
        print(f"Std Ψ(x): {results_tensor[:, 3].std().item():.3f}")
        
        # Analyze correlations
        correlation = torch.corrcoef(results_tensor.T)
        print(f"Correlation matrix shape: {correlation.shape}")
        
        assert results_tensor.shape == (5, 4)

if __name__ == "__main__":
    # Run tests manually for learning
    test_basics = TestPyTorchBasics()
    test_integration = TestFarmerPyTorchIntegration()
    
    print("=== PyTorch Basics ===")
    test_basics.test_tensor_creation()
    print("\n" + "="*50 + "\n")
    
    test_basics.test_neural_network_creation()
    print("\n" + "="*50 + "\n")
    
    test_basics.test_training_loop_basics()
    print("\n" + "="*50 + "\n")
    
    test_basics.test_gradients_and_backprop()
    print("\n" + "="*50 + "\n")
    
    print("=== Farmer-PyTorch Integration ===")
    test_integration.test_neural_accuracy_with_pytorch()
    print("\n" + "="*50 + "\n")
    
    test_integration.test_tensor_operations_for_psi()
