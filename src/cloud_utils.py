import runpod
import os
import json
import requests
import time

class RunPodClient:
    def __init__(self):
        self.api_key = os.getenv('RUNPOD_API_KEY')
        self.api_url = "https://api.runpod.io/graphql"
        
    def execute_graphql(self, query, variables=None):
        """Execute a GraphQL query against RunPod API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json={'query': query, 'variables': variables}
        )
        
        return response.json()
    
    def run_pod_command(self, pod_id, command):
        """Run a command on a pod using the GraphQL API"""
        query = """
        mutation podCommand($input: PodCommandInput!) {
            podCommand(input: $input) {
                output
                error
            }
        }
        """
        
        variables = {
            'input': {
                'podId': pod_id,
                'command': command
            }
        }
        
        result = self.execute_graphql(query, variables)
        return result.get('data', {}).get('podCommand', {})

def sync_to_pod(pod_id=None):
    """Sync files to RunPod using the GraphQL API"""
    client = RunPodClient()
    
    if pod_id is None:
        # Get all pods and find the active one
        pods = runpod.get_pods()
        print("Found pods:", json.dumps(pods, indent=2))
        
        active_pods = [p for p in pods if p.get('desiredStatus') == 'RUNNING']
        if not active_pods:
            print("No active pods found!")
            return False
            
        pod = active_pods[0]
        pod_id = pod['id']
    
    try:
        print(f"\nSyncing to pod {pod_id}...")
        
        # Create directory
        print("Creating directory structure...")
        result = client.run_pod_command(pod_id, "mkdir -p ~/shakespeare")
        if result.get('error'):
            print("Error creating directory:", result['error'])
            return False
            
        # For each file/directory we want to sync
        files_to_sync = ['./data', './src', 'How LLMs Work.ipynb']
        for file_path in files_to_sync:
            if os.path.exists(file_path):
                print(f"Preparing to sync {file_path}...")
                
                # Create tar of the file/directory
                tar_name = f"{os.path.basename(file_path)}.tar.gz"
                os.system(f"tar czf {tar_name} {file_path}")
                
                # Convert tar to base64
                with open(tar_name, 'rb') as f:
                    import base64
                    content = base64.b64encode(f.read()).decode('utf-8')
                
                # Send file content via command
                print(f"Uploading {file_path}...")
                result = client.run_pod_command(pod_id, f"""
                    cd ~/shakespeare && \
                    echo '{content}' | base64 -d > {tar_name} && \
                    tar xzf {tar_name} && \
                    rm {tar_name}
                """)
                
                # Clean up local tar
                os.remove(tar_name)
                
                if result.get('error'):
                    print(f"Error uploading {file_path}:", result['error'])
                    return False
        
        print("Files synced successfully!")
        return True
        
    except Exception as e:
        print(f"Error syncing files: {str(e)}")
        return False

def run_on_pod(command, pod_id=None):
    """Run a command on the pod using the GraphQL API"""
    client = RunPodClient()
    
    if pod_id is None:
        pods = runpod.get_pods()
        active_pods = [p for p in pods if p.get('desiredStatus') == 'RUNNING']
        if not active_pods:
            print("No active pods found!")
            return False
        pod = active_pods[0]
        pod_id = pod['id']
    
    try:
        print(f"Running command on pod {pod_id}: {command}")
        
        if 'cloud_train.py' in command:
            # Extract config and write it to file
            config_str = command.split("'")[1]
            result = client.run_pod_command(pod_id, f"""
                cd ~/shakespeare && \
                cat > config.json << 'EOL'
{config_str}
EOL
            """)
            
            if result.get('error'):
                print("Error writing config:", result['error'])
                return False
            
            # Run the training script
            result = client.run_pod_command(pod_id, 
                "cd ~/shakespeare && python cloud_train.py config.json"
            )
        else:
            result = client.run_pod_command(pod_id, command)
        
        if result.get('error'):
            print("Command error:", result['error'])
            return False
            
        print("Command output:", result.get('output', ''))
        return True
        
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return False