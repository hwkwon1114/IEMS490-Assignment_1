# 1. Start with a standard, lightweight Python base image.
# This is like choosing the chassis or frame for our build.
FROM python:3.10-slim

# 2. Set the working directory inside the container's file system.
# All subsequent commands will run from this '/app' directory.
WORKDIR /app

# 3. Copy only the requirements file first.
# This is an optimization. Docker caches layers, so if requirements.txt
# doesn't change, it won't re-install all the packages on subsequent builds.
COPY requirements.txt .

# 4. Install all the Python packages listed in your requirements.txt.
# --no-cache-dir is a good practice to keep the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all of your project files (the .py scripts, etc.) into the container.
COPY . .

# 6. Set the default command to run when the container starts.
# As documented in your README, we'll start a bash shell to allow
# the user (your instructor) to run each experiment script individually.
CMD ["bash"]
