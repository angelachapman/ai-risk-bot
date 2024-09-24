FROM python:3.9

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy requirements first to cache dependencies
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Command to run the app
CMD ["chainlit", "run", "app.py", "--port", "7860"]

#WORKDIR $HOME/app
#COPY --chown=user . $HOME/app
#COPY ./requirements.txt ~/app/requirements.txt
#RUN pip install -r requirements.txt
#COPY . .
#CMD ["chainlit", "run", "app.py", "--port", "7860"]