The code I have written may still need to be optimized.

The functions in the script include transformation functions that map a point on the undeformed image to a point
on the deformed image. There are three transformation functions with each for affine, similarity, and rigid
transformation. Each of these functions take three arguments. The first argument is the vector that will be
mapped onto the deformed image. The second argument can be a list of tuples which contain pairs of tuples that
represent the mapping of some points on the undeformed image to points on the deformed image. The third argument
is a scalar and can affect how exactly a point maps onto the deformed image. If it is not specified explicitly,
it is set equal to 1 by default.

Example usage
-------------

>>> import mls
>>> v = (4, 5)
>>> mapping = [((1, 3), (2, 5)), ((5, 7), (6, 9)), ((9, 4), (10, 6)), ((8, 8), (9, 10))]
>>> mls.affine_transform(v, mapping)
(5.0, 7.0)
>>> v = (7, 5)
>>> mapping = [((1, 3), (2, 6)), ((5, 7), (6, 8)), ((9, 4), (11, 7)), ((8, 8), (8, 12))]
>>> mls.similarity_transform(v, mapping, 2)
(8.541449549976313, 7.312174324964473)
>>> v = (9, 12)
>>> mapping = [((2, 5), (3, 7)), ((1, 4), (3, 5)), ((5, 7), (7, 9)), ((12, 17), (16, 16)), ((9, 15), (13, 14))]
>>> mls.rigid_transform(v, mapping, 1.5)
(12.083169675230895, 11.352663913628428)
