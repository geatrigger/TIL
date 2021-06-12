# Add remote
```
git remote add origin https://github.com/geatrigger/ToyHomepage.git
```

# Set upstream branch(match local branch, remote branch)
```
git push --set-upstream origin master
```

# Secure store username and password
```
git config --global credential.helper 'cache --timeout=3600'
```
<https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/>
```
// create ssh Github key and save private key in local
// mush type a secure passphrase
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

eval `ssh-agent -s`
ssh-add -k ~/.ssh/id_rsa
// should use remote as ssh
git remote set-url origin git@github.com:geatrigger/ToyHomepage.git
git remote get-url origin
```

# Delete credential cache
```
git config credential.helper store
git config --global --unset credential.helper
```