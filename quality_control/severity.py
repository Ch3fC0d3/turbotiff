ORDER={"info":0,"low":1,"medium":2,"high":3,"critical":4}
def blocks(severity):return ORDER[severity]>=3
