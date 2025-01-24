import sys 
from src.logger import logging



def error_message_details(error,error_detail:sys):
    _,_,exe_tb=error_detail.exc_info()
    file_name=exe_tb.tb_frame.f_code.co_filename 
    error_message="Error occured inpython script [{0}] line number [{1}] error message[{2}]".formate(
    file_name,exe_tb.tb_lineno,str(error))
    
    return error_message

    

class CustomException(Exception):
    def __inti__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_details=error_details)

    def __str__(self):
        return self.error_message


# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Zero Division error occured")
#         raise CustomException(e,sys)