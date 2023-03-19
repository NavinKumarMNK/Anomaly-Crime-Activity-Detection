import os
from multiprocessing.pool import ThreadPool

def run_script(ip, username, password):
    os.system(f'sshpass -p {password} scp -o StrictHostKeyChecking=no script.sh {username}@{ip}:~/')
    os.system(f'sshpass -p {password} ssh -o StrictHostKeyChecking=no {username}@{ip} "chmod +x ~/script.sh && ~/script.sh"')

if __name__ == '__main__':
    # List of IP addresses to run the script on
    ips = ['172.16.96.65']

    # Username and password for SSH authentication
    username = 'windows'
    password = 'win@123'

    # Run the script on all remote servers in parallel
    with ThreadPool(processes=len(ips)) as pool:
        pool.starmap(run_script, [(ip, username, password) for ip in ips])

    print("Script executed on all IP addresses.")

