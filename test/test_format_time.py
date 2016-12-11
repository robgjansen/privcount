from privcount.log import normalise_time, format_epoch, format_period, format_datetime

# some of this code is only executed when we reach 1 week
# so let's test it thoroughly

# this test will always exit successfully, unless the logging code throws
# an exception: any formatting issues need to be identified by inspection

def test_format_time(time):
  print "normalise_time({}) = {}".format(time, normalise_time(time))
  print "format_epoch({}) = {}".format(time, format_epoch(time))
  print "format_period({}) = {}".format(time, format_period(time))
  print "format_datetime({}) = {}".format(time, format_datetime(time))

# negative minute
test_format_time(-61)

# negative second
test_format_time(-1)

# zero
test_format_time(0)

# fractional
test_format_time(0.1)
test_format_time(0.49)
test_format_time(0.5)
test_format_time(0.51)
test_format_time(0.99)

# second
test_format_time(1)
test_format_time(1.1)
test_format_time(32)

# minute
test_format_time(59)
test_format_time(60)
test_format_time(61)
test_format_time(83)
test_format_time(100)
test_format_time(1000)

# hour
test_format_time(3599)
test_format_time(3600)
test_format_time(3601)
test_format_time(3659)
test_format_time(3660)
test_format_time(3661)
test_format_time(10000)

# day
test_format_time(86399)
test_format_time(86400)
test_format_time(86401)
test_format_time(100000)

# week
test_format_time(604799)
test_format_time(604800)
test_format_time(604801)
test_format_time(1000000)
test_format_time(10000000)

# year (52 weeks)
test_format_time(31449599)
test_format_time(31449600)
test_format_time(31449601)

# 53 weeks
test_format_time(32054400)

test_format_time(100000000)
test_format_time(1000000000)

# recently (examples used in util.py)
test_format_time(1468691880)
test_format_time(1469600312)
test_format_time(1469600312 - 1468691880)
test_format_time(1468691880 - 1469600312)

# > 2**32 and 2038
test_format_time(10000000000)
test_format_time(100000000000)

# and past this point it's pretty much implausible
