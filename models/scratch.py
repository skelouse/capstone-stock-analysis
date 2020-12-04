
from functools import partial


def test_func(a, b, apple):
    print(a, b, apple)


func = partial(test_func, apple=5)
loss_func.__name__= "cross_entropy"
print(func.__name__)
func(1, 2)



# loss_func = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=1)

# model.compile(loss=loss_func)
