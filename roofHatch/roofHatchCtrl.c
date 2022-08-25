// DATA TYPES
typedef struct {
   RequestType_T                 Request_Type;
} RoofHatch_Ctrl_In_T;

typedef struct {
    RoofHatch_T                  RoofHatch_Command;
} RoofHatch_Ctrl_Out_T;

// VARIABLES
static RoofHatch_T               m_previousCommand;
static RequestType_T             m_previousRequestType;
static Counter_T                 m_protectionCounter; 

// PRIVATE FUNCTION PROTOTYPES
static void readAllData(RoofHatch_Ctrl_In_T *a_in);
static void writeAllData(const RoofHatch_Ctrl_Out_T *a_out);

// PRIVATE FUNCTIONS
static void readAllData(RoofHatch_Ctrl_In_T *a_in)
{
   Std_ReturnType             readStatus;
   
   readStatus  = Rte_Read_Request_Type_Request_Type(&(a_in->Request_Type));
   if (readStatus != RTE_E_OK  )
   {
         a_in->Request_Type = m_previousRequestType;
   }
  
}

static void writeAllData(const RoofHatch_Ctrl_Out_T *a_out)
{
   (void)Rte_Write_RoofHatch_Command_RoofHatch_Command(a_out->RoofHatch_Command);
}

static void inactivateMotors(RoofHatch_Ctrl_Out_T *a_out)
{
    m_protectionCounter = 0;
    m_previousCommand = NO_COMMAND;
    a_out->RoofHatch_Command = NO_COMMAND;
}

static void activateMotors(RoofHatch_T cmd, RoofHatch_Ctrl_Out_T *a_out)
{
   if(m_previousCommand == cmd){
      if(m_protectionCounter > THRESHOLD){
         m_previousCommand = NO_COMMAND;
         a_out->RoofHatch_Command = NO_COMMAND;   
      }
      else {
         m_protectionCounter++;
         a_out->RoofHatch_Command = m_previousCommand;   
      }
   }
   else {
      if(m_protectionCounter > THRESHOLD){
         a_out->RoofHatch_Command = m_previousCommand;   
      }
      else {
         m_protectionCounter=0;
         m_previousCommand = cmd;
         a_out->RoofHatch_Command = cmd;
      }
   }
}

// PUBLIC FUNCTIONS
FUNC(void, RTE_ROOFHATCHCTRL_APPL_CODE) RoofHatch_Ctrl_Init(void)
{
   m_previousCommand = NO_COMMAND;
   m_protectionCounter = 0;
}

FUNC(void, RTE_ROOFHATCHCTRL_APPL_CODE) RoofHatch_Ctrl_run(void)
{
   RoofHatch_Ctrl_In_T  m_in;
   RoofHatch_Ctrl_Out_T m_out;

   readAllData(&m_in);

   if(m_in.Request_Type == RoofHatch_rqst_Open){
      activateMotors(OPEN,&m_out);
   }
   else if(m_in.Request_Type == RoofHatch_rqst_Close){
      activateMotors(CLOSE,&m_out);
   }
   else {
      inactivateMotors(&m_out)
   }

   writeAllData(&m_out);
}
