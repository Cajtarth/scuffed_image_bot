# Import the command handler
import lightbulb, hikari
import os
from dotenv import load_dotenv
import ai_model
from moviepy.editor import VideoFileClip
import asyncio
import threading
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

logging.getLogger("apscheduler").propagate = False

# Functions intended to be run on a thread

def gen_image_and_save_to_file(prompt, model, filename, neg_prompt, quality, nsfw, rating):
    global queue
    #print("Starting image thread...")
    ai_model.gen_image(prompt, model, filename, neg_prompt, quality, nsfw, rating)
    for q in queue:
        if q["filename"] == filename:
            #print("Marking complete...")
            q["complete"] = True

def gen_video_and_save_to_file(prompt, model, format, filename):
    global queue
    #print("Starting video thread...")
    ai_model.gen_video(prompt, model, format, filename)
    for q in queue:
        if q["filename"] == filename:
            #print("Marking complete...")
            q["complete"] = True

def gen_response(response_queue_index):
    global queue
    global response_queue
    #print("Starting response thread...")

    print(f"Assigning index {response_queue_index} from response_queue of length {len(response_queue)}.")
    i = response_queue_index
    r = response_queue[i]

    r["response"] = ai_model.gen_response_text(r["prompt"], r["owner"], r["type"], r["queue_size"])
    #print("Marking response complete...")
    r["complete"] = True

def refine_image_and_save_to_file(prompt, base_image, filename):
    global queue
    #print("Starting refine thread...")
    ai_model.refine_image(prompt, base_image, filename)
    for q in queue:
        if q["filename"] == filename:
            #print("Marking complete...")
            q["complete"] = True




# Global variables
global queue #Generation queue
queue = []
global backup_queue
backup_queue = []
global response_queue #Queue for text responses (abandoned)
response_queue = []
global currently_working #If the bot is currently working on something
currently_working = False
global previously_drawing #If the bot is in the middle of the queue
previously_drawing = False

global suspended #Used to suspend bot from taking queue requests
suspended = False
global prev_suspended
prev_suspended = False



# Load Discord bot token but keep it a secret in the .env file
# (In .env file, "DISCORD_BOT_TOKEN_SIB=[token here]")
load_dotenv()
bot_secret = os.getenv("DISCORD_BOT_TOKEN_SIB")

# Instantiate a Bot instance
bot = lightbulb.BotApp(token=bot_secret, intents=hikari.Intents.ALL)

# Needed tro schedule check_queue function
sched = AsyncIOScheduler()
sched.start()

# Function to check the queue every 3 seconds
@sched.scheduled_job(CronTrigger(second="*/3"))
async def check_queue():
    global currently_working
    global previously_drawing
    global response_queue
    global queue
    global backup_queue
    global suspended
    global prev_suspended

    #print("Checking queue...")
    if len(queue) == 0 and len(response_queue) == 0:
        #Nothing in any queue so abort
        #await asyncio.sleep(5) #Was using this to sleep but the scheduler didn't seem to like it
        return
    
    if currently_working == False:
        # At this point we know there is at least one thing in some queue so we need to start on the first item
        # When responses were generated, I wanted to prioritize those to improve responsiveness
        currently_working = True
        if len(response_queue) > 0:
            #print("Prioritizing response in queue...")
            r = response_queue[0]
            if r["complete"]:
                await respond_response(0)
                return
            _thread = threading.Thread(target=gen_response, args=([0]))
            _thread.start()
            return
        else:
            #print("No responses in queue...")
            pass
        
        if len(queue) > 0: #and no responses in queue and not currently working
            #print("Starting on next item in queue...")
            q = queue[0]
            if q["complete"]:
                #This shouldn't ever trigger but just in case something is already done...
                #Maybe in the future we can try multithreading if the top x number of requests use the same model?
                if q["type"] == "image":
                    await respond_image_and_update_queue(0)
                    return
                elif q["type"] == "video":
                    await respond_video_and_update_queue(0)
                    return
                elif q["type"] == "refine":
                    await respond_refine_and_update_queue(0)
                    return
                
            if not previously_drawing and suspended:
                return

            #Each type of return has a different set of dictionary values so we have to sort them; they all should have "type"
            #Also each type needs a different function to start a thread with
            if q["type"] == "image":
                if previously_drawing or prev_suspended: #If we are already working, we want to make sure we let the server know we are starting
                    prev_suspended = False
                    mflags = hikari.MessageFlag.NONE
                    if q["private"] != None:
                        mflags = hikari.MessageFlag.EPHEMERAL #Ephemeral is a private message that only the user can see.
                    await q["ctx"].respond(q["last_prompt"], flags=mflags) #("Starting on image with prompt....")
                previously_drawing = True
                _thread = threading.Thread(target=gen_image_and_save_to_file, args=(q["ctx"].options.prompt, q["model"], q["filename"], q["neg_prompt"], q["quality"], q["nsfw"], q["rating"]))
                _thread.start()
                return
            elif q["type"] == "video":
                if previously_drawing:
                    await q["ctx"].respond(q["last_prompt"])
                previously_drawing = True
                _thread = threading.Thread(target=gen_video_and_save_to_file, args=(q["ctx"].options.prompt, q["model"], q["format"], q["filename"]))
                _thread.start()
                return
            elif q["type"] == "refine":
                if previously_drawing:
                    await q["ctx"].respond(q["last_prompt"])
                previously_drawing = True
                _thread = threading.Thread(target=refine_image_and_save_to_file, args=(q["ctx"].options.prompt, q["base_image"], q["filename"]))
                _thread.start()
                return

    # At this point, there should be something in queue and currently_working should be true so we just need to cycle through everything and check if it's done
    if len(response_queue) > 0:
        #print(f"Prioritizing responses ({len(response_queue)})...")
        for i in range(len(response_queue)):
            r = response_queue[i]
            if r["complete"] == True:
                await respond_response(i)
                return
        #Fall through (no return) in case a response request was added after an image to make sure that we don't deadlock
    
    if len(queue) > 0:
        for i in range(len(queue)):
            if queue[i]["complete"] == True:
                if queue[i]["type"] == "image":
                    await respond_image_and_update_queue(i)
                    break
                elif queue[i]["type"] == "video":
                    await respond_video_and_update_queue(i)
                    break
                elif queue[i]["type"] == "refine":
                    await respond_refine_and_update_queue(i)
                    break


#Respond functions all assume that the generation is done at the provided queue_index
#They are all similar and build the message before replying, popping the queue, and setting currently_working/previously_drawing if needed
async def respond_refine_and_update_queue(queue_index):
    global queue
    global currently_working
    global previously_drawing

    i = queue_index
    filename = queue[i]["filename"]
    ctx = queue[i]["ctx"]

    f = hikari.File(filename)

    red_x_emoji = "❌"
    row = bot.rest.build_message_action_row()
    button = row.add_interactive_button(hikari.ButtonStyle.PRIMARY, "delete", emoji=red_x_emoji)

    await ctx.respond(queue[i]["prompt"], attachments=[f], component=row)

    queue.pop(i)

    if len(queue) == 0:
        previously_drawing = False
    currently_working = False


async def respond_response(queue_index):
    global response_queue
    global currently_working

    ctx = response_queue[queue_index]["ctx"]

    await ctx.respond(response_queue[queue_index]["response"])

    response_queue.pop(queue_index)
    currently_working = False

async def respond_image_and_update_queue(queue_index):
    global queue
    global currently_working
    global previously_drawing

    i = queue_index
    filename = queue[i]["filename"]
    ctx = queue[i]["ctx"]

    f = hikari.File(filename)

    red_x_emoji = "❌"
    row = bot.rest.build_message_action_row()
    button = row.add_interactive_button(hikari.ButtonStyle.PRIMARY, "delete", emoji=red_x_emoji)

    if queue[i]["private"] == None:
        await ctx.respond(queue[i]["prompt"], attachments=[f], component=row)
    else:
        #Send in a DM if private is set
        a = ctx.author

        dmc = await a.fetch_dm_channel()
        await dmc.send(queue[i]["prompt"], attachments=[f], component=row)

    queue.pop(i)
    if len(queue) == 0:
        previously_drawing = False
    currently_working = False

async def respond_video_and_update_queue(queue_index):
    global queue
    global currently_working
    global previously_drawing

    f = ""
    q = queue[queue_index]
    ctx = q["ctx"]
    filename = q["filename"]
    format = q["format"]

    #I couldn't get save_to_gif to work so instead I just convert to a GIF from an MP4 if requested

    if format == ai_model.MP4:
        "Waiting for file to exist..."
        f = filename+"."+format
        timer = 0
        while not os.path.exists(f):
            #This is old code that shouldn't activate but I had some weird race conditions with the generation of the file and os.exists
            if timer > 5:
                print("I'm blacking out!")
                return
            timer += 1
            await asyncio.sleep(1)
    elif format == ai_model.GIF:
        f = "files/temp/temp.mp4"
        #print("Waiting for file for gif conversion...")
        timer = 0
        while not os.path.exists(f):
            #See above in MP4 branch
            if timer > 5:
                print("I'm blacking out!")
                return
            timer += 1
            await asyncio.sleep(1)
        videoClip = VideoFileClip(f)
        videoClip.write_gif(filename+"."+format)
    if os.path.exists(f):
        f = hikari.File(f)

        red_x_emoji = "❌"
        row = bot.rest.build_message_action_row()
        button = row.add_interactive_button(hikari.ButtonStyle.PRIMARY, "delete", emoji=red_x_emoji)

        await ctx.respond(q["prompt"], attachments=[f], component=row)
        queue.pop(queue_index)
        if len(queue) == 0:
            previously_drawing = False
        currently_working = False



@bot.listen()
async def on_start(event: hikari.events.lifetime_events.StartedEvent) -> None:
    #Plan to let the bot announce that it is online/offline eventually
    pass


@bot.listen()
async def on_message_reply(event: hikari.events.message_events.DMMessageCreateEvent) -> None:
    #Old way to delete a bot message (replying to the image you want to delete with "delete"); not used anymore
    message = event.message
    author = message.author
    reacted_message = message.referenced_message

    if reacted_message == None:
        return
    if reacted_message.author.id == bot.get_me().id:
        if message.content == "delete":
            print("Deleting message...")
            await reacted_message.delete()

@bot.listen()
async def on_component_interaction(event: hikari.InteractionCreateEvent) -> None:
    #New way to delete a bot image; listen for the 'delete' id on the buttons in the responses
    if not isinstance(event.interaction, hikari.ComponentInteraction):
        return
    
    if event.interaction.custom_id == "delete":
        await event.interaction.create_initial_response(hikari.ResponseType.DEFERRED_MESSAGE_UPDATE)
        await event.interaction.message.delete()




@bot.command
#@lightbulb.option("reload", "Don't use this. It does nothing.", required=False, choices=["yes", "yes but no"])
@lightbulb.option("neg_prompt", "[optional] Negative prompt to use for Animagine or Proteus ONLY.", required=False)
@lightbulb.option("private", "Makes the generated image private to you and DMs you the result.", required=False, choices=["True"])
@lightbulb.option("nsfw", "LEAVE BLANK/UNFILLED FOR SFW (NSFW will show up as a black box)", required=False, choices=["True"])
@lightbulb.option("rating", "[optional] Rating to use for Animagine ONLY.", required=False, choices=["General", "Sensitive", "Questionable/NSFW", "Explicit/NSFW"])
@lightbulb.option("quality", "[optional] Quality to use for Animagine ONLY. WARNING: Top 2 qualities may produce NSFW!", required=False, choices=["worst quality", "low quality", "normal quality", "medium quality", "high quality", "best quality", "masterpiece"])
@lightbulb.option("prompt", "Prompt to use for image generation")
@lightbulb.option("model", "Model to use for image generation", choices=["Midjourney", "Stable Diffusion", "Proteus", "Animagine", "Orange"])
@lightbulb.command("scuff", "Generates an image based on a prompt.")
@lightbulb.implements(lightbulb.SlashCommand)
async def scuff(ctx: lightbulb.Context) -> None:
    #This probably needs to be cleaned up as it has gotten quite bloated. I didn't plan to have so many models with so many different options!
    global currently_working
    global queue
    global backup_queue
    global suspended

    '''
    if suspended:
        await ctx.respond("Bot currently suspended. Please unsuspend bot or try again later.")
        return
    '''

    #Convert plaintext names to bot models (I hate string comparisons and they crop up too much in this whole thing)
    lmodel = ""
    if ctx.options.model == "Midjourney":
        lmodel = ai_model.MIDJOURNEY
    elif ctx.options.model == "Stable Diffusion":
        lmodel = ai_model.STABLE_DIFFUSION
    elif ctx.options.model == "Animagine":
        lmodel = ai_model.ANIMAGINE
    elif ctx.options.model == "Proteus":
        lmodel = ai_model.PROTEUS
    elif ctx.options.model == "Orange":
        lmodel = ai_model.ORANGE

    mflags = hikari.MessageFlag.NONE
    if ctx.options.private != None:
        mflags = hikari.MessageFlag.EPHEMERAL

    
    #Response needs to be within 3 seconds to work with discord
    await ctx.respond(ctx.author.username + " requested a drawing of \"" + ctx.options.prompt +
                    "\" (using " + ctx.options.model + ") from Scuffed Image Bot.", flags=mflags)
    
    last_prompt = "Starting on " + ctx.author.username + "'s image with prompt " + ctx.options.prompt
    
    neg_prompt = ctx.options.neg_prompt
    quality = ctx.options.quality
    rating = ctx.options.rating

    #I erringly thought that these would be blank strings but they are None so I have to convert them
    if neg_prompt == None:
        neg_prompt = ""
    if quality == None:
        quality = ""
    if rating == None:
        rating = ""

    time = ctx.event.interaction.created_at
    #Unique filename; in future may be able to assign unique id to allow for easier recall
    filename = "files/" + ctx.author.username + "-" + str(time.year) + "-" + str(time.month) + "-" + str(time.day) + "-" + str(time.hour) + "-" + str(time.minute) + "-" + str(time.second) + ".png"
    
    #Not using responses anymore
    #response_queue.append({"response":"", "type":"image", "complete":False, "ctx":ctx, "prompt":ctx.options.prompt, "owner":ctx.author.username, "queue_size":len(queue)})

    #await ai_model.gen_image(ctx.options.prompt, lmodel, filename, neg_prompt, quality) #Slow and locked up the bot while generating
    
    #if not currently_working:
        #pass
        #Instead of doing this manually, we will let the queue handle it
        #'''currently_working = True
        #_thread = threading.Thread(target=gen_image_and_save_to_file, args=(ctx.options.prompt, lmodel, filename, neg_prompt, quality))
        #_thread.start()'''
    #else:
    if currently_working or suspended:
        #Need to let the users know if a request is being queued (possibly privately)
        mflags = hikari.MessageFlag.NONE
        if ctx.options.private != None:
            mflags = hikari.MessageFlag.EPHEMERAL
        if suspended:
            await ctx.respond("Queuing " + ctx.author.username +"'s request to be completed after bot suspension is lifted...", flags=mflags)
        else:
            await ctx.respond("Queuing " + ctx.author.username +"'s request...", flags=mflags)

    dict = {"type":"image", 
            "complete":False, 
            "ctx":ctx, 
            "last_prompt":last_prompt, 
            "prompt": ctx.options.prompt,
            "model":lmodel, 
            "filename":filename, 
            "neg_prompt":neg_prompt, 
            "rating":rating, 
            "quality":quality, 
            "owner":ctx.author.username, 
            "nsfw":ctx.options.nsfw, 
            "private":ctx.options.private}
    
    if suspended:
        backup_queue.append(dict)
    else:
        #Append request to queue in form of dictionary; this probably should have been a full fledged class but it made sense when I started...
        queue.append(dict)
    
        
@bot.command
@lightbulb.option("format", "Format to use for generated file.", required=False, choices=[ai_model.MP4, ai_model.GIF])
@lightbulb.option("model", "Model to use for video generation.", choices=["Zeroscope (general)", "Animov (anime)"])
@lightbulb.option("prompt", "Prompt to use for video generation.")
@lightbulb.command("video-gen", "Generates video based on a prompt.", auto_defer=True)
@lightbulb.implements(lightbulb.SlashCommand)
async def video_gen(ctx: lightbulb.Context) -> None:
    #See scuff above as these functions are very similar
    global currently_working
    global queue
    global backup_queue
    global suspended

    if suspended:
        await ctx.respond("Bot currently suspended. Please unsuspend bot or try again later.")
        return

    if ctx.options.format == "test":
        f = hikari.File("files/test.mp4")
        await ctx.respond(f)
        return
    
    model = ""
    if ctx.options.model == "Zeroscope (general)":
        model = ai_model.ZEROSCOPE
    elif ctx.options.model == "Animov (anime)":
        model = ai_model.ANIMOV

    if model == "":
        model = ai_model.ZEROSCOPE
    
    format = ""
    if ctx.options.format == None:
        format = ai_model.MP4
    else:
        format = ctx.options.format

    await ctx.respond(ctx.author.username + " requested a " + format + " of \"" + ctx.options.prompt +
            "\" (using " + ctx.options.model + ") from Scuffed Image Bot.")
    
    time = ctx.event.interaction.created_at
    filename = "files/" + ctx.author.username + "-" + str(time.year) + "-" + str(time.month) + "-" + str(time.day) + "-" + str(time.hour) + "-" + str(time.minute) + "-" + str(time.second)

    #await ai_model.gen_video(ctx.options.prompt, model, format, filename) #Slow and locked up the bot while generating
    last_prompt = "Starting on " + ctx.author.username + "'s " + format + " with prompt " + ctx.options.prompt

    #response_queue.append({"response":"", "type":"video", "complete":False, "ctx":ctx, "prompt":ctx.options.prompt, "owner":ctx.author.username, "queue_size":len(queue)})

    if not currently_working:
        pass
        '''currently_working = True
        _thread = threading.Thread(target=gen_video_and_save_to_file, args=(ctx.options.prompt, model, format, filename))
        _thread.start()'''
    else:
        await ctx.respond("Queuing " + ctx.author.username +"'s request...")
    queue.append({"type":"video", "complete":False, "ctx":ctx, "last_prompt":last_prompt, "prompt": ctx.options.prompt,"model":model, "filename":filename, "format":format, "owner":ctx.author.username, "private":None})
    


@bot.command
@lightbulb.option("type", "Sets if the debug text is from a video or image.", choices=["image", "video"])
@lightbulb.option("prompt", "Prompt to use for text generation")
@lightbulb.command("text-gen", "Generates text based on a prompt.")
@lightbulb.implements(lightbulb.SlashCommand)
async def text_gen(ctx: lightbulb.Context) -> None:
    #Debug function to test out text responses but I kind of gave up on them.
    global queue
    global suspended

    #message = (ctx.author.username + " used prompt \"" + ctx.options.prompt +
    #                "\" to generate text.")
    #await ctx.respond(message)
    if suspended:
        await ctx.respond("Bot currently suspended. Please unsuspend bot or try again later.")
        return

    await ctx.respond(f"Debug function used to add prompt {ctx.options.prompt} to the response queue...")

    response_queue.append({"response":"", "type":ctx.options.type, "complete":False, "ctx":ctx, "prompt":ctx.options.prompt, "owner":ctx.author.username, "queue_size":len(queue)})

    #response = await ai_model.gen_text(ctx.options.prompt, ctx.author.username, len(queue),_system_prompt=ctx.options.system_prompt)
    #await ctx.respond(response)

@bot.command
@lightbulb.option("prompt", "Prompt to use for text generation")
@lightbulb.option("base_image", "Sets the base image to refine.")
@lightbulb.command("scuff-refine", "Refines an image based on a prompt.")
@lightbulb.implements(lightbulb.SlashCommand)
async def scuff_refine(ctx: lightbulb.Context) -> None:
    #See scuff above as these functions are very similar
    global queue
    global backup_queue
    global suspended

    if suspended:
        await ctx.respond("Bot currently suspended. Please unsuspend bot or try again later.")
        return
    
    await ctx.respond(f"{ctx.author.username} requested refinement of an image ({ctx.options.base_image}) of {ctx.options.prompt} from Scuffed Image Bot.")
    
    time = ctx.event.interaction.created_at
    filename = "files/" + ctx.author.username + "-" + str(time.year) + "-" + str(time.month) + "-" + str(time.day) + "-" + str(time.hour) + "-" + str(time.minute) + "-" + str(time.second) + ".png"
    last_prompt = "Starting on " + ctx.author.username + "'s refinement with prompt " + ctx.options.prompt

    #An actually fairly nicely formatted dictionary! I should probably reformat the other two functions...
    queue.append({"type":"refine", 
                 "complete":False, 
                 "ctx":ctx, 
                 "last_prompt":last_prompt, 
                 "prompt": ctx.options.prompt,
                 "base_image":ctx.options.base_image, 
                 "filename":filename,
                 "owner":ctx.author.username,
                 "private":None})

    #response = await ai_model.gen_text(ctx.options.prompt, ctx.author.username, len(queue),_system_prompt=ctx.options.system_prompt)
    #await ctx.respond(response)



@bot.command
@lightbulb.command("queue", "Displays the current queue.")
@lightbulb.implements(lightbulb.SlashCommand)
async def display_queue(ctx: lightbulb.Context) -> None:
    #Function to display the current queue as a numbered list and skip over privated items
    global queue
    global suspended

    if not suspended or len(queue) > 0 or len(backup_queue) > 0:
        formatted_queue = ""
        if len(queue) > 0:
            if suspended:
                formatted_queue = "**(Bot Suspended) Current Queue:\n"
            else:
                formatted_queue = "**Current Queue:\n"
            formatted_queue += "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬**\n"
            formatted_queue += "\n".join(
                (f"**{i+1}.** {item['ctx'].author.username} - {item['type']} - {item['prompt']}" if item["private"] is None
                else f"")
                for i,item in enumerate(queue)
            )
            formatted_queue += "\n\n"

            #await ctx.respond(formatted_queue)
        else:
            #await ctx.respond("No current queue items.")
            formatted_queue += "No current queue items. \n\n"

        if len(backup_queue) > 0:
            formatted_queue += "**Backlog Queue:\n"
            formatted_queue += "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬**\n"
            formatted_queue += "\n".join(
                (f"**{i+1}.** {item['ctx'].author.username} - {item['type']} - {item['prompt']}" if item["private"] is None
                else f"")
                for i,item in enumerate(backup_queue)
            )
            formatted_queue += "\n"
        
        await ctx.respond(formatted_queue)
    else:
        await ctx.respond("Bot currently suspended with no items in queue.")

@bot.command
@lightbulb.command("suspend", "Toggles suspending the bot from accepting requests.")
@lightbulb.implements(lightbulb.SlashCommand)
async def suspend(ctx: lightbulb.Context) -> None:
    #Suspends bot which restricts any further queues. Unfortunately the bot still uses a lot of memory if any models are loaded.
    global queue
    global backup_queue
    global suspended
    global prev_suspended

    if not suspended:
        await ctx.respond("Suspending bot...")
        suspended = True
    else:
        await ctx.respond("Resuming bot function...")
        suspended = False
        prev_suspended = True
        if len(backup_queue) > 0:
            await ctx.respond("Starting on backlog of requests...")
            for i in range(len(backup_queue)):
                queue.append(backup_queue.pop(0))

@bot.command
@lightbulb.command("debug", "Debug for testing.")
@lightbulb.implements(lightbulb.SlashCommand)
async def debug(ctx: lightbulb.Context) -> None:
    pass

#Finally, let's run the bot :D
bot.run()