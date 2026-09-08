from datetime import datetime

from main import main
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

@scheduler.scheduled_job('interval', minutes=15,next_run_time= datetime.now(), id='scheduled_job')
def scheduled_job():
    main()

if __name__ == "__main__":
    scheduler.start()