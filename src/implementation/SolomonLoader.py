class SolomonLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load_data(self, max_customers=10):
        """Load Solomon instance data, limiting to max_customers"""
        try:
            # Read all lines from file
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
            
            # Strip whitespace and empty lines
            lines = [line.strip() for line in lines if line.strip()]
            
            # Find vehicle capacity
            vehicle_capacity = None
            for line in lines:
                if "VEHICLE" in line.upper() and "CAPACITY" in line.upper():
                    try:
                        # Try to find the last number in the line
                        parts = line.split()
                        for part in reversed(parts):
                            if part.isdigit():
                                vehicle_capacity = int(part)
                                break
                    except:
                        continue
            
            if not vehicle_capacity:
                print("Warning: Could not find vehicle capacity, using default 200")
                vehicle_capacity = 200
            
            # Find customer data section
            start_idx = None
            for i, line in enumerate(lines):
                # Look for common headers in Solomon datasets
                if any(header in line.upper() for header in ["CUST NO.", "CUSTNO", "NODE_COORD_SECTION"]):
                    start_idx = i + 1
                    break
            
            if start_idx is None:
                # If no header found, try to find where the customer data begins
                for i, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) >= 7 and all(self._is_number(p) for p in parts[:7]):
                        start_idx = i
                        break
            
            if start_idx is None:
                raise ValueError("Could not find customer data section")
            
            # Parse customer data
            customers = []
            for line in lines[start_idx:]:
                parts = line.split()
                if len(parts) >= 7 and all(self._is_number(p) for p in parts[:7]):
                    customers.append({
                        'id': int(float(parts[0])),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'demand': float(parts[3]),
                        'ready_time': float(parts[4]),
                        'due_time': float(parts[5]),
                        'service_time': float(parts[6])
                    })
                    if len(customers) > max_customers:
                        break
            
            if not customers:
                raise ValueError("No valid customer data found")
            
            print(f"Successfully loaded {len(customers)} customers")
            print(f"Vehicle capacity: {vehicle_capacity}")
            
            return customers, vehicle_capacity
            
        except Exception as e:
            print(f"Error loading Solomon data: {str(e)}")
            return None, None
    
    def _is_number(self, s):
        """Check if string can be converted to float"""
        try:
            float(s)
            return True
        except ValueError:
            return False

# Test function for the loader
def test_solomon_loader(file_path, max_customers=5):
    print(f"Testing Solomon loader with file: {file_path}")
    loader = SolomonLoader(file_path)
    customers, capacity = loader.load_data(max_customers)
    
    if customers and capacity:
        print("\nFirst customer data:")
        print(customers[0])
        print(f"\nVehicle capacity: {capacity}")
        print(f"Total customers loaded: {len(customers)}")
        return True
    return False

if __name__ == "__main__":
    # Test the loader first
    file_path = "/content/c101.txt"
    success = test_solomon_loader(file_path)
    
    if success:
        # Create environment with the loaded data
        env = SolomonVRPEnvironment(file_path, max_customers=5)
        
        # Test a few random steps
        state = env.reset()
        print("\nInitial state shape:", state.shape)
        
        # Show initial configuration
        env.render()
        
        # Try a few random actions
        for i in range(3):
            action = np.random.randint(0, env.num_customers)
            state, reward, done = env.step(action)
            print(f"\nStep {i+1}:")
            print(f"Action: {action}")
            print(f"Reward: {reward:.2f}")
            print(f"Done: {done}")
        
        # Show final configuration
        env.render()
