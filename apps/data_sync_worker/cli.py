import argparse
import logging
import os
import apps.data_sync_worker.deployment as deployment
# from apps.data_sync_worker.main import run_synchronization # Example import

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    deployment.main()

if __name__ == "__main__":
    main()