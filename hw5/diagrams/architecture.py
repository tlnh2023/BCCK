"""Architecture diagram generation"""

from diagrams import Diagram, Cluster
from diagrams.onprem.queue import Kafka
from diagrams.onprem.database import MongoDB, Cassandra
from diagrams.onprem.analytics import Spark
from diagrams.onprem.vcs import Github
from diagrams.onprem.social import Reddit
from diagrams.onprem.aggregator import Fluentd
from diagrams.onprem.inmemory import Redis


def create_architecture_diagram():
    """Create and save the architecture diagram"""
    
    with Diagram("Social Media Storage Architecture", show=False, filename="architecture"):
        
        with Cluster("Data Sources"):
            github = Github("GitHub Issues")
            reddit = Reddit("Reddit Posts")
        
        with Cluster("Data Collection"):
            crawlers = Fluentd("Crawlers")
        
        with Cluster("Stream Processing"):
            kafka = Kafka("Kafka")
            spark = Spark("Spark Processor")
        
        with Cluster("Storage Layer"):
            mongodb = MongoDB("MongoDB\n(Documents)")
            cassandra = Cassandra("Cassandra\n(Time-Series)")
        
        # Define data flow
        [github, reddit] >> crawlers >> kafka >> spark
        spark >> [mongodb, cassandra]
    
    print("Architecture diagram saved as 'architecture.png'")


if __name__ == "__main__":
    create_architecture_diagram()
