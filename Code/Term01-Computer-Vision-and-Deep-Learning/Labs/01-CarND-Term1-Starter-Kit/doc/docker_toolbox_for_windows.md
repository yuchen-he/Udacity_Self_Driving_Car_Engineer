# Docker Toolbox for Windows
### CarND Programming Environment - Installation Instructions

## Step 1: Download Docker and Run Installer

[Download](https://github.com/docker/toolbox/releases/download/v1.12.3/DockerToolbox-1.12.3.exe)
Docker Toolbox for Windows

Make sure to install VirtualBox with the NDIS5 driver.

## Step 2: `docker run hello-world`

This step ensures Docker is installed properly on your computer.

Launch Docker Quickstart Terminal and type the following at the prompt.
This will implicitly pull and then and run the `hello-world` Docker image
(https://hub.docker.com/_/hello-world/) in a new container:

```sh
$ docker run hello-world
```

You should now see the following confirming successful installation:

```
docker run hello-world
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
c04b14da8d14: Pull complete
Digest: sha256:0256e8a36e2070f7bf2d0b0763dbabdd67798512411de4cdcf9431a1feb60fd9
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.
```

That's it!!

**Note** If you see an error like this

```sh
Error creating machine: Error in driver during machine creation:
This computer doesn't have VT-X/AMD-v enabled.
```

this means that you will need to enable virtualization for your computer. Follow
these [instructions](http://www.howtogeek.com/213795/how-to-enable-intel-vt-x-in-your-computers-bios-or-uefi-firmware/).
