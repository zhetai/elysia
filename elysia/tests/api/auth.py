import unittest

# class TestAuth(unittest.TestCase):

#     def test_increment_user_requests(self):

#         from elysia.api.services.user import UserManager

#         user_manager = UserManager()
#         user_manager.remove_user_db("test_user")
#         user_manager.add_user_db("test_user")

#         try:
#             user_manager.get_user_db("test_user")
#         except Exception as e:
#             self.assertTrue(True) # expect an error

#         # brand new user
#         user = user_manager.get_user_db("test_user")
#         self.assertEqual(user["num_requests"], 0)

#         # increment user requests
#         user_manager.increment_user_requests_db("test_user")
#         user = user_manager.get_user_db("test_user")
#         self.assertEqual(user["num_requests"], 1)

#         # reset user requests
#         user_manager.reset_user_requests_db("test_user")
#         user = user_manager.get_user_db("test_user")
#         self.assertEqual(user["num_requests"], 0)

#         # exceed max requests
#         for _ in range(user_manager.max_requests_per_day):
#             user_manager.increment_user_requests_db("test_user")
#         user = user_manager.get_user_db("test_user")
#         self.assertTrue(user_manager.check_user_exceeds_max_requests("test_user"))
