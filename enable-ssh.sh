#!/bin/bash
password=win@123
while read host; do
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no windows@"$host" "echo '$password' | sudo -S systemctl enable ssh"
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no windows@"$host" "echo '$password' | sudo -S systemctl start ssh"

done < host_list.txt

