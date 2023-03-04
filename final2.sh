#!/bin/bash

# List of IP addresses to open Firefox on
ips=("172.16.96.68" "172.16.96.67" "172.16.96.59")

# Loop through the list of IPs and open Firefox on each one
for ip in "${ips[@]}"
do
    sshpass -p "win@123" ssh -X windows@$ip ifconfig &
done

echo "Firefox has been opened on all IP addresses."

