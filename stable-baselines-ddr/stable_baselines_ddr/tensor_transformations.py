import tensorflow as tf


def repeat_inner_dim(input: tf.Tensor, repeats: tf.Tensor) -> tf.Tensor:
    """
    Take a 2D tensor eg [[1,2],[3,4]] and repeat the inside n times eg
    [[1,2],[3,4],[1,2],[3,4],......]
    Args:
        input: A 2D tensor
        repeats: Number of times to repeat (a tf scalar)

    Returns:
        A tensor with the elements of the inner dimension repeated
    """
    minus_one = tf.constant(-1)
    inner_size = tf.shape(input)[1]
    reshape_tensor = tf.stack([minus_one, inner_size])

    expanded = tf.expand_dims(input, axis=0)
    repeated = tf.repeat(expanded, repeats, axis=0)
    reshaped_inner = tf.reshape(repeated, reshape_tensor)
    return reshaped_inner


def repeat_outer_dim(input: tf.Tensor, repeats: tf.Tensor) -> tf.Tensor:
    """
    Take a 1D tensor eg [1,2,3] and repeat its content n times eg
    [1,2,3,1,2,3,1,2,3.......]
    Args:
        input: A 1D tensor
        repeats: Number of times to repead (a tf scalar)

    Returns:
        A tensor with the elements repeated
    """
    expanded = tf.expand_dims(input, axis=0)
    repeated = tf.repeat(expanded, repeats, axis=0)
    reshaped_inner = tf.reshape(repeated, [-1])
    return reshaped_inner
