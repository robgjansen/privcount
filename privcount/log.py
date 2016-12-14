'''
Created on Dec 6, 2016

@author: teor
'''

import logging
import sys
import traceback

from time import time, strftime, gmtime

def log_error():
    _, _, tb = sys.exc_info()
    #traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    logging.warning("An error occurred in file '%s', at line %d, in func %s, in statement '%s'", filename, line, func, text)

## Logging: Time Formatting Functions ##
## a timestamp is an absolute point in time, in seconds since unix epoch
## a period is a relative time duration, in seconds
## a time argument is either a period or a timestamp
## a desc argument is a string description of the timestamp's meaning
## All period and timestamp arguments are normalised using normalise_time()
## before any calculations or formatting are performed

def normalise_time(time):
    '''
    Return the normalised value of time
    An abstraction used for consistent time rounding behaviour
    '''
    # we ignore microseconds
    return int(time)

def current_time():
    '''
    Return the normalised value of the current time
    '''
    return normalise_time(time())

def format_period(period):
    '''
    Format a time period as a human-readable string
    period is in seconds
    Returns a string of the form:
    1w 3d 12h 20m 32s
    starting with the first non-zero period (seconds are always included)
    '''
    period = normalise_time(period)
    period_str = ""
    # handle negative times by prepending a minus sign
    if period < 0:
        period_str += "-"
        period = -period
    # there's no built-in way of formatting a time period like this in python.
    # strftime is almost there, but would have issues with year-long periods.
    # divmod gives us the desired floor division result, and the remainder,
    # which will be floating point if normalise_time() returns floating point
    (week,   period) = divmod(period, 7*24*60*60)
    (day,    period) = divmod(period,   24*60*60)
    (hour,   period) = divmod(period,      60*60)
    (minute, period) = divmod(period,         60)
    # if normalise_time yields floating point values (microseconds), this will
    # produce a floating point result, which will be formatted as NN.NN
    # if it's an integer, it will format as NN. This is the desired behaviour.
    second           =        period % (      60)
    # now build the formatted string starting with the first non-zero period
    larger_period = 0
    if week > 0:
        period_str += "{}w ".format(week)
        larger_period = 1
    if day > 0 or larger_period:
        period_str += "{}d ".format(day)
        larger_period = 1
    if hour > 0 or larger_period:
        period_str += "{}h ".format(hour)
        larger_period = 1
    if minute > 0 or larger_period:
        period_str += "{}m ".format(minute)
    # seconds are always included, even if they are zero, or if there is no
    # larger period
    period_str += "{}s".format(second)
    return period_str

def format_datetime(timestamp):
    '''
    Format a timestamp as a human-readable UTC date and time string
    timestamp is in seconds since the epoch
    Returns a string of the form:
    2016-07-16 17:58:00
    '''
    timestamp = normalise_time(timestamp)
    return strftime("%Y-%m-%d %H:%M:%S", gmtime(timestamp))

def format_epoch(timestamp):
    '''
    Format a timestamp as a unix epoch numeric string
    timestamp is in seconds since the epoch
    Returns a string of the form:
    1468691880
    '''
    timestamp = normalise_time(timestamp)
    return str(timestamp)

def format_time(period, desc, timestamp):
    '''
    Format a period and timestamp as a human-readable string in UTC
    period is in seconds, and timestamp is in seconds since the epoch
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 1468691880)
    '''
    return "{} ({} {} {})".format(format_period(period),
                                  desc,
                                  format_datetime(timestamp),
                                  format_epoch(timestamp))

def format_interval(period, desc, begin_timestamp, end_timestamp):
    '''
    Format a period and two interval timestamps as a human-readable string in UTC
    period is in seconds, and the timestamps are in seconds since the epoch
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 to 2016-07-27 06:18:32,
    1468691880 to 1469600312)
    '''
    return "{} ({} {} to {}, {} to {})".format(format_period(period),
                                               desc,
                                               format_datetime(begin_timestamp),
                                               format_datetime(end_timestamp),
                                               format_epoch(begin_timestamp),
                                               format_epoch(end_timestamp))

def format_elapsed_time_wait(elapsed_period, desc):
    '''
    Format the time elapsed since a past event, and the past event time in UTC
    elapsed_period is in seconds
    The event time is the current time minus elapsed_period
    elapsed_period is typically time_since_checkin, and desc is typically 'at'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-27 06:18:32 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    elapsed_period = normalise_time(elapsed_period)
    past_timestamp = current_time() - elapsed_period
    return format_time(elapsed_period, desc, past_timestamp)

def format_elapsed_time_since(past_timestamp, desc):
    '''
    Format the time elapsed since a past event, and that event's time in UTC
    past_timestamp is in seconds since the epoch
    The elapsed time is from past_timestamp to the current time
    past_timestamp is typically status['time'], and desc is typically 'since'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 1468691880)
    '''
    # Normalise before calculation to avoid truncation errors
    past_timestamp = normalise_time(past_timestamp)
    elapsed_period = current_time() - past_timestamp
    return format_time(elapsed_period, desc, past_timestamp)

def format_delay_time_wait(delay_period, desc):
    '''
    Format the time delay until a future event, and the expected event time
    in UTC
    delay_period is in seconds
    The event time is the current time plus delay_period
    delay_period is typically config['defer_time'], and desc is typically 'at'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-27 06:18:32 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    delay_period = normalise_time(delay_period)
    future_timestamp = current_time() + delay_period
    return format_time(delay_period, desc, future_timestamp)

def format_delay_time_until(future_timestamp, desc):
    '''
    Format the time delay until a future event, and the expected event time
    in UTC
    The time delay is the difference between future_timestamp and the current
    time
    future_timestamp is in seconds since the epoch
    future_timestamp is typically config['defer_time'], and desc is typically 'at'
    returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-27 06:18:32 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    future_timestamp = normalise_time(future_timestamp)
    delay_period = future_timestamp - current_time()
    return format_time(delay_period, desc, future_timestamp)

def format_interval_time_between(begin_timestamp, desc, end_timestamp):
    '''
    Format the interval elapsed between two events, and the times of those
    events in UTC
    The timestamps are in seconds since the epoch
    The interval is between begin_time and end_time
    desc is typically 'from'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 to 2016-07-27 06:18:32,
    1468691880 to 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    begin_timestamp = normalise_time(begin_timestamp)
    end_timestamp = normalise_time(end_timestamp)
    period = end_timestamp - begin_timestamp
    return format_interval(period, desc, begin_timestamp, end_timestamp)

def format_last_event_time_since(last_event_timestamp):
    '''
    Format the time elapsed since the last Tor event, and that event's time
    in UTC
    last_event_timestamp is in seconds since the epoch, and can be None
    for no events
    The elapsed time is from last_event_timestamp to the current time
    Returns a string in one of the following forms:
    no Tor events received
    last Tor event was 1w 3d 12h 20m 32s (at 2016-07-16 17:58:00 1468691880)
    '''
    if last_event_timestamp is None:
        return "no Tor events received"
    else:
        return "last Tor event was {}".format(format_elapsed_time_since(
                                                  last_event_timestamp, 'at'))
