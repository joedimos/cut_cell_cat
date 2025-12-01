import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from verified_simulator import main_verified, VerifiedCategoricalSimulator
    import numpy as np
    
    def test_installation():
        """Test if all dependencies are available"""
        print("Testing installation...")
        
        # Test numpy
        test_array = np.array([1, 2, 3])
        print("✓ NumPy working")
        
        # Test our classes
        try:
            # Try to create a simple simulator instance
            sim = VerifiedCategoricalSimulator(resolution=10)
            print("✓ Verified simulator initialized")
            return True
        except Exception as e:
            print(f"✗ Error initializing simulator: {e}")
            return False
    
    def main():
        print("Categorical Cut-Cell System - Installation Check")
        print("=" * 50)
        
        if test_installation():
            print("\n" + "=" * 50)
            print("Starting main simulation...")
            print("=" * 50)
            main_verified()
        else:
            print("\nPlease check the installation instructions above.")
            sys.exit(1)
            
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nPlease make sure:")
    print("1. All Python files are in the same directory")
    print("2. Required packages are installed: numpy, matplotlib")
    print("3. You're using Python 3.7+")
    print("\nInstall with: pip install numpy matplotlib")
    sys.exit(1)

if __name__ == "__main__":
    main()